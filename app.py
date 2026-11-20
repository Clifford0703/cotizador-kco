import io
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cotizador KCO Group", layout="wide")


# ==========================================
# 1. PARÁMETROS CONTRACTUALES Y OPERATIVOS
# ==========================================
@dataclass
class ParametrosKCO:
    rebate: float = 0.065               # 6.5% sobre venta
    cd_acopio: float = 0.012            # 1.2% sobre venta
    promo_5_cobros: float = 0.0085      # 0.85% sobre venta (5 cobros/año)
    activar_promo: bool = False
    alquiler_jaba: float = 0.14         # S/ jaba día
    dias_alquiler: int = 1
    saneo: float = 1.00                 # S/ jaba (si se activa)
    activar_saneo: bool = False
    peso_jaba: float = 18.0             # kg por jaba
    seleccion: float = 0.15             # S/kg
    transporte: float = 0.10            # S/kg
    camara: float = 0.10                # S/kg
    merma: float = 0.0                  # % compra
    utilidad_meta: float = 1.00         # S/kg vendido
    portal_b2b: float = 150.0           # S/ mes
    ventas_mes_est: float = 20000.0     # kg/mes
    tardanza: float = 0.0               # S/ jaba día
    dias_tardanza: int = 0

    @property
    def tasa_descuentos(self) -> float:
        return self.rebate + self.cd_acopio + (self.promo_5_cobros if self.activar_promo else 0.0)

    @property
    def factor_neto_venta(self) -> float:
        return 1.0 - self.tasa_descuentos

    @property
    def jaba_skg(self) -> float:
        total_jaba = (
            self.alquiler_jaba * self.dias_alquiler
            + (self.saneo if self.activar_saneo else 0.0)
            + self.tardanza * self.dias_tardanza
        )
        return total_jaba / self.peso_jaba if self.peso_jaba > 0 else 0.0

    @property
    def operativos_skg(self) -> float:
        return self.seleccion + self.transporte + self.camara + self.jaba_skg

    @property
    def fijo_b2b_skg(self) -> float:
        return self.portal_b2b / self.ventas_mes_est if self.ventas_mes_est > 0 else 0.0


def tabla_lote_vacia() -> pd.DataFrame:
    return pd.DataFrame({
        "Código": pd.Series(dtype="str"),
        "Base Compra": pd.Series(dtype="float64"),
        "Base Proyectado": pd.Series(dtype="float64"),
        "Precio Ofertado": pd.Series(dtype="float64"),
        "Objetivo SPSA": pd.Series(dtype="float64"),
        "Kilos Compra": pd.Series(dtype="float64"),
    })


if "tabla_lote" not in st.session_state:
    st.session_state.tabla_lote = tabla_lote_vacia()


# ==========================================
# 2. MOTOR DE CÁLCULO UNITARIO Y DE LOTE
# ==========================================
def cotizar_item(
    base_compra: Optional[float],
    precio_ofertado: Optional[float],
    objetivo_spsa: Optional[float],
    kilos_compra: Optional[float],
    p: ParametrosKCO,
) -> Optional[dict]:
    if base_compra is None or pd.isna(base_compra) or base_compra <= 0:
        return None

    factor_merma = 1.0 - p.merma
    k_neto = p.factor_neto_venta
    if factor_merma <= 0 or k_neto <= 0:
        return None

    costo_base_real = base_compra / factor_merma
    costos_fijos_op = p.operativos_skg + p.fijo_b2b_skg

    precio_equilibrio = (costo_base_real + costos_fijos_op) / k_neto
    precio_min_meta = (costo_base_real + costos_fijos_op + p.utilidad_meta) / k_neto

    tiene_oferta = precio_ofertado is not None and not pd.isna(precio_ofertado) and precio_ofertado > 0
    if tiene_oferta:
        rebate_skg = precio_ofertado * p.rebate
        cd_skg = precio_ofertado * p.cd_acopio
        promo_skg = precio_ofertado * (p.promo_5_cobros if p.activar_promo else 0.0)
        costo_total_skg = costo_base_real + costos_fijos_op + rebate_skg + cd_skg + promo_skg
        utilidad_skg = precio_ofertado - costo_total_skg
        margen_venta_pct = (utilidad_skg / precio_ofertado) * 100
        estado = "CUMPLE" if utilidad_skg >= p.utilidad_meta - 1e-6 else "REVISAR"
    else:
        rebate_skg = cd_skg = promo_skg = costo_total_skg = None
        utilidad_skg = margen_venta_pct = None
        estado = "PENDIENTE OFERTA"

    tiene_kilos = kilos_compra is not None and not pd.isna(kilos_compra) and kilos_compra > 0
    kilos_vendibles = kilos_compra * factor_merma if tiene_kilos else None
    utilidad_total = (kilos_vendibles * utilidad_skg) if (tiene_kilos and tiene_oferta) else None
    venta_total = (kilos_vendibles * precio_ofertado) if (tiene_kilos and tiene_oferta) else None

    return {
        "costo_base_real": costo_base_real,
        "operativos_skg": p.operativos_skg,
        "fijo_b2b_skg": p.fijo_b2b_skg,
        "precio_equilibrio": precio_equilibrio,
        "precio_min_meta": precio_min_meta,
        "rebate_skg": rebate_skg,
        "cd_skg": cd_skg,
        "promo_skg": promo_skg,
        "costo_total_skg": costo_total_skg,
        "utilidad_skg": utilidad_skg,
        "margen_venta_pct": margen_venta_pct,
        "kilos_vendibles": kilos_vendibles,
        "utilidad_total": utilidad_total,
        "venta_total": venta_total,
        "estado": estado,
    }


def optimizar_compensacion_lote(
    df_base: pd.DataFrame,
    p: ParametrosKCO,
    modo: str = "Proporcional a Kilos (tope SPSA)",
    tope_subida_skg: Optional[float] = None,
    usar_base_proy: bool = False,
) -> Optional[dict]:
    """
    Evalúa el lote completo y distribuye la brecha de utilidad faltante entre códigos
    con margen disponible hasta el Objetivo SPSA (o tope configurado).
    """
    k_neto = p.factor_neto_venta
    factor_merma = 1.0 - p.merma
    if k_neto <= 0 or factor_merma <= 0:
        return None

    registros = []
    for _, row in df_base.iterrows():
        base_col = "Base Proyectado" if (usar_base_proy and pd.notna(row["Base Proyectado"]) and row["Base Proyectado"] > 0) else "Base Compra"
        base_val = row[base_col]
        oferta_val = row["Precio Ofertado"]
        kilos_val = row["Kilos Compra"]
        spsa_val = row["Objetivo SPSA"]

        if pd.isna(base_val) or base_val <= 0 or pd.isna(oferta_val) or oferta_val <= 0 or pd.isna(kilos_val) or kilos_val <= 0:
            continue

        costo_base_real = base_val / factor_merma
        costos_fijos_op = p.operativos_skg + p.fijo_b2b_skg
        kg_vend = kilos_val * factor_merma
        u_actual = oferta_val * k_neto - (costo_base_real + costos_fijos_op)
        u_tot_actual = kg_vend * u_actual
        u_meta_tot = kg_vend * p.utilidad_meta
        venta_actual = kg_vend * oferta_val

        # Límite superior de precio para compensar: Objetivo SPSA si existe, o sin límite adicional
        tiene_spsa = pd.notna(spsa_val) and spsa_val > 0
        techo_spsa = float(spsa_val) if tiene_spsa else np.inf
        techo_tope = (float(oferta_val) + float(tope_subida_skg)) if (tope_subida_skg is not None and tope_subida_skg > 0) else np.inf
        precio_techo = max(float(oferta_val), min(techo_spsa, techo_tope))
        max_subida_posible = max(0.0, precio_techo - float(oferta_val))
        capacidad_u_extra = kg_vend * max_subida_posible * k_neto
        holgura_spsa = max(0.0, float(spsa_val) - float(oferta_val)) if tiene_spsa else 0.0

        registros.append({
            "Código": row["Código"] if pd.notna(row["Código"]) else "",
            "Base Usada": base_val,
            "Costo Base Real": costo_base_real,
            "Costos Fijos+Op": costos_fijos_op,
            "Kilos Vendibles": kg_vend,
            "Precio Ofertado": float(oferta_val),
            "Objetivo SPSA": float(spsa_val) if tiene_spsa else np.nan,
            "Precio Techo": precio_techo,
            "Holgura SPSA": holgura_spsa,
            "Max Subida": max_subida_posible,
            "Capacidad U Extra": capacidad_u_extra,
            "Utilidad Actual (S/kg)": u_actual,
            "Utilidad Actual Total": u_tot_actual,
            "Utilidad Meta Total": u_meta_tot,
            "Venta Actual Total": venta_actual,
        })

    if not registros:
        return None

    df = pd.DataFrame(registros)
    u_actual_lote = df["Utilidad Actual Total"].sum()
    u_meta_lote = df["Utilidad Meta Total"].sum()
    venta_actual_lote = df["Venta Actual Total"].sum()
    brecha_lote = u_meta_lote - u_actual_lote

    df["Precio Compensado"] = df["Precio Ofertado"].copy()

    if brecha_lote > 1e-6:
        restante_u = brecha_lote
        # Iteración agua-llenado respetando techos individuales
        activos = df[df["Max Subida"] > 1e-9].copy()
        if not activos.empty:
            if modo == "Priorizar holgura vs. SPSA":
                # Ordena de mayor holgura relativa hacia menor para cargar primero donde hay más espacio
                activos = activos.sort_values(by=["Holgura SPSA", "Kilos Vendibles"], ascending=[False, False])
                for idx in activos.index:
                    if restante_u <= 1e-6:
                        break
                    cap_item = df.loc[idx, "Capacidad U Extra"]
                    aporte = min(restante_u, cap_item)
                    delta_p = aporte / (df.loc[idx, "Kilos Vendibles"] * k_neto)
                    df.loc[idx, "Precio Compensado"] = df.loc[idx, "Precio Ofertado"] + delta_p
                    restante_u -= aporte
            else:
                # Proporcional a kilos vendibles (con reparto iterativo si algún código toca su techo)
                pendiente_idx = list(activos.index)
                while restante_u > 1e-6 and pendiente_idx:
                    peso_total = df.loc[pendiente_idx, "Kilos Vendibles"].sum()
                    if peso_total <= 0:
                        break
                    algun_tope = False
                    for idx in list(pendiente_idx):
                        peso_frac = df.loc[idx, "Kilos Vendibles"] / peso_total
                        aporte_prop = restante_u * peso_frac
                        max_restante_item = (df.loc[idx, "Precio Techo"] - df.loc[idx, "Precio Compensado"]) * df.loc[idx, "Kilos Vendibles"] * k_neto
                        if aporte_prop >= max_restante_item - 1e-9:
                            df.loc[idx, "Precio Compensado"] = df.loc[idx, "Precio Techo"]
                            restante_u -= max(0.0, max_restante_item)
                            pendiente_idx.remove(idx)
                            algun_tope = True
                    if not algun_tope:
                        for idx in pendiente_idx:
                            peso_frac = df.loc[idx, "Kilos Vendibles"] / peso_total
                            aporte_prop = restante_u * peso_frac
                            delta_p = aporte_prop / (df.loc[idx, "Kilos Vendibles"] * k_neto)
                            df.loc[idx, "Precio Compensado"] += delta_p
                        restante_u = 0.0

    df["Ajuste Precio (S/kg)"] = df["Precio Compensado"] - df["Precio Ofertado"]
    df["Utilidad Comp. (S/kg)"] = df["Precio Compensado"] * k_neto - (df["Costo Base Real"] + df["Costos Fijos+Op"])
    df["Margen Comp. (%)"] = (df["Utilidad Comp. (S/kg)"] / df["Precio Compensado"]) * 100
    df["Utilidad Comp. Total"] = df["Kilos Vendibles"] * df["Utilidad Comp. (S/kg)"]
    df["Venta Comp. Total"] = df["Kilos Vendibles"] * df["Precio Compensado"]

    u_comp_lote = df["Utilidad Comp. Total"].sum()
    venta_comp_lote = df["Venta Comp. Total"].sum()
    margen_pond_actual = (u_actual_lote / venta_actual_lote * 100) if venta_actual_lote > 0 else 0.0
    margen_pond_comp = (u_comp_lote / venta_comp_lote * 100) if venta_comp_lote > 0 else 0.0
    brecha_restante = max(0.0, u_meta_lote - u_comp_lote)

    return {
        "tabla": df,
        "u_actual_lote": u_actual_lote,
        "u_meta_lote": u_meta_lote,
        "u_comp_lote": u_comp_lote,
        "brecha_lote": brecha_lote,
        "brecha_restante": brecha_restante,
        "margen_pond_actual": margen_pond_actual,
        "margen_pond_comp": margen_pond_comp,
    }


# ==========================================
# 3. BARRA LATERAL: PARÁMETROS CONTRACTUALES
# ==========================================
with st.sidebar:
    st.header("Parámetros KCO")
    utilidad_meta = st.number_input("Utilidad meta (S/kg)", value=1.00, min_value=0.0, step=0.05)
    merma_pct = st.number_input("Merma (%)", value=0.0, min_value=0.0, max_value=40.0, step=0.5) / 100

    with st.expander("Acuerdos SPSA (%)"):
        rebate = st.number_input("Rebate (%)", value=6.5, step=0.1) / 100
        cd_acopio = st.number_input("CD Acopio (%)", value=1.2, step=0.1) / 100
        activar_promo = st.checkbox("Activar Promoción 5 cobros (0.85%)", value=False)

    with st.expander("Operativos y Jabas (S/)"):
        seleccion = st.number_input("Selección (S/kg)", value=0.15, step=0.01)
        transporte = st.number_input("Transporte/flete (S/kg)", value=0.10, step=0.01)
        camara = st.number_input("Cámara / Otros (S/kg)", value=0.10, step=0.01)
        peso_jaba = st.number_input("Peso jaba (kg)", value=18.0, min_value=1.0, step=1.0)
        alquiler_jaba = st.number_input("Alquiler jaba (S/día)", value=0.14, step=0.01)
        dias_alquiler = st.number_input("Días alquiler jaba", value=1, min_value=0, step=1)
        activar_saneo = st.checkbox("Activar Saneo (S/ 1.00/jaba)", value=False)

    with st.expander("Fijo B2B y Contingencias"):
        portal_b2b = st.number_input("Portal B2B (S/mes)", value=150.0, min_value=0.0, step=10.0)
        ventas_mes_est = st.number_input("Volumen mensual real (kg)", value=20000.0, min_value=1.0, step=1000.0)
        tardanza = st.number_input("Tardanza (S/jaba día)", value=0.0, min_value=0.0, step=0.50)
        dias_tardanza = st.number_input("Días tardanza", value=0, min_value=0, step=1)

params = ParametrosKCO(
    rebate=rebate,
    cd_acopio=cd_acopio,
    activar_promo=activar_promo,
    alquiler_jaba=alquiler_jaba,
    dias_alquiler=int(dias_alquiler),
    activar_saneo=activar_saneo,
    peso_jaba=peso_jaba,
    seleccion=seleccion,
    transporte=transporte,
    camara=camara,
    merma=merma_pct,
    utilidad_meta=utilidad_meta,
    portal_b2b=portal_b2b,
    ventas_mes_est=ventas_mes_est,
    tardanza=tardanza,
    dias_tardanza=int(dias_tardanza),
)


# ==========================================
# 4. INTERFAZ PRINCIPAL DE LA CALCULADORA
# ==========================================
st.title("Calculadora Cotizadora — KCO Group")
st.caption(
    f"Descuentos SPSA: {params.tasa_descuentos*100:.2f}% | "
    f"Operativos + Jaba: S/ {params.operativos_skg:.4f}/kg | "
    f"Fijo B2B: S/ {params.fijo_b2b_skg:.4f}/kg | "
    f"Ítems en lote: {len(st.session_state.tabla_lote.dropna(how='all'))}"
)

tab_calc, tab_lote = st.tabs(["Calculadora Rápida", "Cotización por Lote"])

# --- PESTAÑA 1: CALCULADORA RÁPIDA ---
with tab_calc:
    c0, c1, c2, c3, c4 = st.columns([1, 1.2, 1.2, 1.2, 1.2])
    codigo_in = c0.text_input("Código / Calibre", value="", placeholder="Ej. A, P, M...")
    base_in = c1.number_input("Base Compra (S/kg)", value=None, min_value=0.0, step=0.10)
    ofertado_in = c2.number_input("Precio Ofertado (S/kg)", value=None, min_value=0.0, step=0.05)
    spsa_in = c3.number_input("Objetivo SPSA (S/kg)", value=None, min_value=0.0, step=0.10)
    kilos_in = c4.number_input("Kilos Compra (kg)", value=None, min_value=0.0, step=100.0)

    res = cotizar_item(base_in, ofertado_in, spsa_in, kilos_in, params)

    if res is None:
        st.info("Ingresa al menos **Base Compra (S/kg)** para calcular a cuánto debes vender.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio Equilibrio (U = S/ 0.00)", f"S/ {res['precio_equilibrio']:.2f}/kg")
        m2.metric(f"Precio Mínimo Meta (U = S/ {params.utilidad_meta:.2f})", f"S/ {res['precio_min_meta']:.2f}/kg")
        m3.metric(
            "Utilidad al Precio Ofertado",
            f"S/ {res['utilidad_skg']:.2f}/kg" if res["utilidad_skg"] is not None else "—",
            f"Margen {res['margen_venta_pct']:.1f}%" if res["margen_venta_pct"] is not None else "Ingresa oferta",
        )
        m4.metric(
            "Utilidad Total del Lote",
            f"S/ {res['utilidad_total']:,.2f}" if res["utilidad_total"] is not None else "—",
            f"{res['kilos_vendibles']:,.1f} kg vendibles" if res["kilos_vendibles"] is not None else "Ingresa kilos",
        )

        if st.button("Agregar a Cotización por Lote", type="primary"):
            nueva_fila = pd.DataFrame([{
                "Código": codigo_in.strip(),
                "Base Compra": float(base_in),
                "Base Proyectado": None,
                "Precio Ofertado": float(ofertado_in) if ofertado_in is not None else None,
                "Objetivo SPSA": float(spsa_in) if spsa_in is not None else None,
                "Kilos Compra": float(kilos_in) if kilos_in is not None else None,
            }])
            st.session_state.tabla_lote = pd.concat(
                [st.session_state.tabla_lote, nueva_fila],
                ignore_index=True,
            )
            st.success(f"Agregado a la cotización por lote (total: {len(st.session_state.tabla_lote)} ítems).")

        if res["utilidad_skg"] is not None:
            desglose = pd.DataFrame([
                {"Concepto": "Precio Ofertado", "S/ por kg": ofertado_in, "% Venta": 100.0},
                {"Concepto": "(-) Costo Base ajustado por merma", "S/ por kg": -res["costo_base_real"], "% Venta": -(res["costo_base_real"] / ofertado_in) * 100},
                {"Concepto": "(-) Rebate SPSA", "S/ por kg": -res["rebate_skg"], "% Venta": -params.rebate * 100},
                {"Concepto": "(-) CD Acopio", "S/ por kg": -res["cd_skg"], "% Venta": -params.cd_acopio * 100},
                {"Concepto": "(-) Promoción 5 cobros", "S/ por kg": -res["promo_skg"], "% Venta": -(params.promo_5_cobros * 100 if params.activar_promo else 0.0)},
                {"Concepto": "(-) Operativos (Sel + Flete + Cám + Jaba)", "S/ por kg": -res["operativos_skg"], "% Venta": -(res["operativos_skg"] / ofertado_in) * 100},
                {"Concepto": "(-) Fijo Portal B2B", "S/ por kg": -res["fijo_b2b_skg"], "% Venta": -(res["fijo_b2b_skg"] / ofertado_in) * 100},
                {"Concepto": "(=) UTILIDAD NETA", "S/ por kg": res["utilidad_skg"], "% Venta": res["margen_venta_pct"]},
            ])
            st.dataframe(
                desglose.style.format({"S/ por kg": "S/ {:.4f}", "% Venta": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )

# --- PESTAÑA 2: COTIZACIÓN POR LOTE ---
with tab_lote:
    st.markdown(
        "**Cómo usar esta tabla:** "
        "1) Agrega ítems desde **Calculadora Rápida** o con **`+`**. "
        "2) Completa **Base Compra**, **Base Proyectado**, **Precio Ofertado**, **Objetivo SPSA** y **Kilos Compra**. "
        "3) Revisa la evaluación individual y luego el **Optimizador de Utilidad del Lote** para compensar precios cuando un competidor apriete un código."
    )

    if st.button("Limpiar lote"):
        st.session_state.tabla_lote = tabla_lote_vacia()
        st.rerun()

    edited_df = st.data_editor(
        st.session_state.tabla_lote,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_lote",
        column_config={
            "Código": st.column_config.TextColumn("Código"),
            "Base Compra": st.column_config.NumberColumn("Base Compra (S/kg)", min_value=0.0, step=0.05, format="S/ %.2f"),
            "Base Proyectado": st.column_config.NumberColumn("Base Proy. (S/kg)", min_value=0.0, step=0.05, format="S/ %.2f"),
            "Precio Ofertado": st.column_config.NumberColumn("Precio Ofertado (S/kg)", min_value=0.0, step=0.05, format="S/ %.2f"),
            "Objetivo SPSA": st.column_config.NumberColumn("Objetivo SPSA (S/kg)", min_value=0.0, step=0.05, format="S/ %.2f"),
            "Kilos Compra": st.column_config.NumberColumn("Kilos Compra", min_value=0.0, step=10.0, format="%.1f"),
        },
    )
    st.session_state.tabla_lote = edited_df

    filas_salida = []
    for _, row in edited_df.iterrows():
        r_real = cotizar_item(row["Base Compra"], row["Precio Ofertado"], row["Objetivo SPSA"], row["Kilos Compra"], params)
        if r_real is None:
            continue
        r_proy = cotizar_item(row["Base Proyectado"], row["Precio Ofertado"], row["Objetivo SPSA"], row["Kilos Compra"], params)
        filas_salida.append({
            "Código": row["Código"] if pd.notna(row["Código"]) else "",
            "Base Compra": row["Base Compra"],
            "Base Proy.": row["Base Proyectado"],
            "P. Equilibrio (U=0)": r_real["precio_equilibrio"],
            "Precio Ofertado": row["Precio Ofertado"],
            "P. Mínimo Meta": r_real["precio_min_meta"],
            "Objetivo SPSA": row["Objetivo SPSA"],
            "Kilos Compra": row["Kilos Compra"],
            "Utilidad Real (S/kg)": r_real["utilidad_skg"],
            "Margen Real (%)": r_real["margen_venta_pct"],
            "Utilidad Proy. (S/kg)": r_proy["utilidad_skg"] if r_proy else None,
            "Utilidad Total (S/)": r_real["utilidad_total"],
            "Estado": r_real["estado"],
        })

    if not filas_salida:
        st.info("Agrega un ítem desde **Calculadora Rápida** o en la tabla superior con al menos **Base Compra**.")
    else:
        df_out = pd.DataFrame(filas_salida)
        st.dataframe(
            df_out.style.format({
                "Base Compra": "S/ {:.2f}",
                "Base Proy.": lambda x: f"S/ {x:.2f}" if pd.notna(x) else "—",
                "P. Equilibrio (U=0)": "S/ {:.2f}",
                "Precio Ofertado": lambda x: f"S/ {x:.2f}" if pd.notna(x) else "—",
                "P. Mínimo Meta": "S/ {:.2f}",
                "Objetivo SPSA": lambda x: f"S/ {x:.2f}" if pd.notna(x) else "—",
                "Kilos Compra": lambda x: f"{x:,.1f}" if pd.notna(x) else "—",
                "Utilidad Real (S/kg)": lambda x: f"S/ {x:.2f}" if pd.notna(x) else "—",
                "Margen Real (%)": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
                "Utilidad Proy. (S/kg)": lambda x: f"S/ {x:.2f}" if pd.notna(x) else "—",
                "Utilidad Total (S/)": lambda x: f"S/ {x:,.2f}" if pd.notna(x) else "—",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # --- OPTIMIZADOR DE COMBINACIONES ENTRE CÓDIGOS Y COSTOS ---
        st.divider()
        st.subheader("Combinaciones y Compensación de Utilidad en el Lote")
        st.caption(
            "Calcula cómo compensar la utilidad del lote completo cuando un competidor baja el precio de un código, "
            "ajustando precios en códigos con margen hasta el Objetivo SPSA o evaluando Base Proyectado."
        )

        o1, o2, o3 = st.columns(3)
        modo_comp = o1.selectbox(
            "Criterio de compensación entre códigos",
            ["Proporcional a Kilos (tope SPSA)", "Priorizar holgura vs. SPSA"],
        )
        tope_subida = o2.number_input("Tope máximo de subida por código (S/kg, 0 = sin tope)", value=0.0, min_value=0.0, step=0.05)
        usar_base_proy = o3.checkbox("Evaluar usando Base Proyectado (donde exista)", value=False)

        opt = optimizar_compensacion_lote(
            edited_df,
            params,
            modo=modo_comp,
            tope_subida_skg=tope_subida if tope_subida > 0 else None,
            usar_base_proy=usar_base_proy,
        )

        if opt is None:
            st.info("Completa **Base Compra**, **Precio Ofertado** y **Kilos Compra** en al menos una fila para calcular la compensación del lote.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Utilidad Actual del Lote", f"S/ {opt['u_actual_lote']:,.2f}", f"Margen pond. {opt['margen_pond_actual']:.1f}%")
            k2.metric(f"Utilidad Meta (S/ {params.utilidad_meta:.2f}/kg)", f"S/ {opt['u_meta_lote']:,.2f}", f"Brecha: S/ {opt['brecha_lote']:,.2f}")
            k3.metric("Utilidad Compensada", f"S/ {opt['u_comp_lote']:,.2f}", f"Margen pond. {opt['margen_pond_comp']:.1f}%")
            k4.metric("Brecha sin cubrir", f"S/ {opt['brecha_restante']:,.2f}", "Lote en meta" if opt["brecha_restante"] <= 0.01 else "Falta techo SPSA")

            df_comp_show = opt["tabla"][[
                "Código", "Base Usada", "Precio Ofertado", "Objetivo SPSA",
                "Ajuste Precio (S/kg)", "Precio Compensado", "Kilos Vendibles",
                "Utilidad Comp. (S/kg)", "Margen Comp. (%)", "Utilidad Comp. Total"
            ]]
            st.dataframe(
                df_comp_show.style.format({
                    "Base Usada": "S/ {:.2f}",
                    "Precio Ofertado": "S/ {:.2f}",
                    "Objetivo SPSA": lambda x: f"S/ {x:.2f}" if pd.notna(x) else "—",
                    "Ajuste Precio (S/kg)": "+S/ {:.2f}" if False else lambda x: f"{'+' if x > 0 else ''}S/ {x:.2f}",
                    "Precio Compensado": "S/ {:.2f}",
                    "Kilos Vendibles": "{:,.1f}",
                    "Utilidad Comp. (S/kg)": "S/ {:.2f}",
                    "Margen Comp. (%)": "{:.1f}%",
                    "Utilidad Comp. Total": "S/ {:,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Cotizacion_Lote")
            if opt is not None:
                opt["tabla"].to_excel(writer, index=False, sheet_name="Compensacion_Lote")
        st.download_button(
            label="Descargar Cotización y Compensación (.xlsx)",
            data=buffer.getvalue(),
            file_name="KCO_Cotizacion_Completa.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
