"""Calculadora de contracción térmica y dimensional - tubería PE 80 / PE 100.

Ejecutar:  pip install streamlit pandas numpy altair
           streamlit run app_contraccion_pe.py
"""
import math
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from packaging.version import Version

st.set_page_config(page_title="Contracción de tubería PE", page_icon="📏", layout="wide")

STRETCH = ({"width": "stretch"} if Version(st.__version__) >= Version("1.50.0")
           else {"use_container_width": True})

# Valores típicos (editar con la ficha técnica del proveedor de resina).
# alpha en 10^-4 1/K  (=  mm/m·K / 1000 * 10^4 -> 1.8 equivale a 0.18 mm/m·K)
MATERIALS = {
    "PE 80": {"rho": 950.0, "alpha": 2.0},
    "PE 100": {"rho": 959.0, "alpha": 1.8},
    "Personalizado": {"rho": 955.0, "alpha": 1.9},
}

st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1250px}
[data-testid="stMetric"]{background:rgba(128,128,128,.08);border:1px solid rgba(128,128,128,.25);
  border-radius:12px;padding:14px 16px}
[data-testid="stMetricValue"]{font-size:clamp(1.25rem,2.3vw,1.85rem)}
.status{border-radius:12px;padding:14px 18px;border-left:8px solid;margin:.6rem 0 1rem}
.status h4{margin:0 0 4px 0}.status p{margin:0;opacity:.92}
.ok{background:rgba(46,125,50,.14);border-color:#2e7d32}
.warn{background:rgba(237,108,2,.15);border-color:#ed6c02}
.bad{background:rgba(211,47,47,.16);border-color:#d32f2f}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------- MODELO --
def mean_wall_temp(t_ext, t_int, d_ext, e, mode):
    """Temperatura media de la pared (°C)."""
    simple = (t_ext + t_int) / 2
    if mode == "Promedio simple":
        return simple
    ro, ri = d_ext / 2, d_ext / 2 - e
    if ri <= 0 or abs(t_ext - t_int) < 1e-9:
        return simple
    ln = math.log(ro / ri)
    # Conducción radial estacionaria, media ponderada por área (r dr)
    f = (ro**2 / 2 * ln - (ro**2 - ri**2) / 4) / ((ro**2 - ri**2) / 2) / ln
    return t_int + (t_ext - t_int) * f


def classify(d, dmin, davg, dmax):
    band = dmax - dmin
    if d < dmin:
        return "bad", "⛔ FUERA DE NORMA · por debajo del mínimo", (
            f"Ø final {d:.2f} mm es {dmin - d:.2f} mm menor al mínimo ({dmin:.2f} mm). "
            "Tubería subdimensionada: aumente el Ø de calibración/extrusión.")
    if d > dmax:
        return "bad", "⛔ FUERA DE NORMA · por encima del máximo", (
            f"Ø final {d:.2f} mm excede en {d - dmax:.2f} mm el máximo ({dmax:.2f} mm). "
            "Tubería sobredimensionada: reduzca el Ø de calibración/extrusión.")
    margin = min(d - dmin, dmax - d)
    if band > 0 and margin < 0.15 * band:
        lado = "mínimo" if d - dmin < dmax - d else "máximo"
        return "warn", "⚠️ DENTRO DE NORMA · cerca del límite", (
            f"Ø final {d:.2f} mm cumple, pero solo a {margin:.2f} mm del {lado}. "
            "Poca holgura ante variaciones de proceso.")
    return "ok", "✅ DENTRO DE NORMA", (
        f"Ø final {d:.2f} mm (desvío vs. promedio {d - davg:+.2f} mm). "
        f"Holgura mínima a los límites: {margin:.2f} mm.")


def compute(p, alpha=None):
    a = (p["alpha"] if alpha is None else alpha) * 1e-4
    tm = mean_wall_temp(p["t_ext"], p["t_int"], p["d0"], p["e0"], p["mode"])
    dT = tm - p["t_final"]
    k_th = 1 - a * dT                          # factor térmico (L_f / L_0)
    k_L = k_th * (1 - p["rev"] / 100)          # + retracción residual opcional
    L_f = p["L_final"] * 1000
    L_exact = L_f / k_L
    d_f, e_f = p["d0"] * k_th, p["e0"] * k_th
    mass_pm = p["rho"] * math.pi * e_f * (d_f - e_f) * 1e-6
    state = classify(d_f, p["dmin"], p["davg"], p["dmax"])
    return dict(a=a, tm=tm, dT=dT, k_th=k_th, k_L=k_L, L_f=L_f, L_exact=L_exact,
                L_rec=L_exact + p["margin"], shrink_mm=L_exact - L_f,
                shrink_pct=(1 - k_L) * 100, d_f=d_f, e_f=e_f, di_f=d_f - 2 * e_f,
                sdr=d_f / e_f, mass_pm=mass_pm, mass_tot=mass_pm * p["L_final"], state=state)


# ---------------------------------------------------------------- ENTRADAS --
def apply_preset():
    m = st.session_state["material"]
    if m != "Personalizado":
        st.session_state["rho"] = MATERIALS[m]["rho"]
        st.session_state["alpha"] = MATERIALS[m]["alpha"]


if "rho" not in st.session_state:
    st.session_state.update(material="PE 100", rho=959.0, alpha=1.8)

with st.sidebar:
    st.header("⚙️ Parámetros")
    st.subheader("1 · Material")
    material = st.selectbox("Resina", list(MATERIALS), key="material", on_change=apply_preset)
    custom = material == "Personalizado"
    rho = st.number_input("Densidad (kg/m³)", 800.0, 1100.0, key="rho", step=1.0,
                          format="%.1f", disabled=not custom)
    alpha = st.number_input("Coef. dilatación lineal α (×10⁻⁴ 1/K)", 0.5, 5.0, key="alpha",
                            step=0.05, format="%.2f", disabled=not custom,
                            help="1.8 ×10⁻⁴ 1/K = 0.18 mm/(m·K). Use la ficha técnica de su resina.")
    st.subheader("2 · Geometría")
    d0 = st.number_input("Ø exterior inicial (mm)", 10.0, 2000.0, 110.7, 0.1, format="%.2f")
    e0 = st.number_input("Espesor de pared promedio (mm)", 1.0, 200.0, 6.6, 0.1, format="%.2f")
    st.subheader("3 · Temperaturas")
    t_ext = st.number_input("Temperatura externa actual (°C)", -20.0, 250.0, 35.0, 1.0)
    t_int = st.number_input("Temperatura interna actual (°C)", -20.0, 250.0, 55.0, 1.0)
    t_final = st.number_input("Temperatura final de contracción (°C)", -20.0, 100.0, 23.0, 1.0)
    st.subheader("4 · Metraje")
    L_final = st.number_input("Metraje final esperado (m)", 0.1, 500.0, 12.0, 0.1, format="%.3f")
    margin = st.number_input("Margen de seguridad de corte (mm)", 0.0, 500.0, 10.0, 1.0,
                             help="Se suma a la longitud exacta (tolerancia de sierra, medición).")
    st.subheader("5 · Límites normativos Ø exterior (mm)")
    dmin = st.number_input("Mínimo", 10.0, 2000.0, 110.0, 0.1, format="%.2f")
    davg = st.number_input("Promedio / nominal", 10.0, 2000.0, 110.2, 0.1, format="%.2f")
    dmax = st.number_input("Máximo", 10.0, 2000.0, 110.4, 0.1, format="%.2f")
    st.caption("Ingrese los valores de la norma aplicable (p. ej. NTC 4585 / ISO 4427).")
    with st.expander("Opciones avanzadas"):
        mode = st.radio("Temperatura media de pared",
                        ["Promedio simple", "Ponderado por área (conducción radial)"])
        rev = st.number_input("Retracción residual adicional (%)", 0.0, 5.0, 0.0, 0.1,
                              help="Reversión longitudinal por orientación molecular (ISO 2505, "
                                   "límite típico ≤ 3 %). No es térmica; 0 si no aplica.")

errors = []
if not dmin < davg < dmax:
    errors.append("Los límites deben cumplir: mínimo < promedio < máximo.")
if e0 >= d0 / 2:
    errors.append("El espesor de pared debe ser menor que el radio exterior.")
if errors:
    for er in errors:
        st.error(er)
    st.stop()

P = dict(rho=rho, alpha=alpha, d0=d0, e0=e0, t_ext=t_ext, t_int=t_int, t_final=t_final,
         L_final=L_final, margin=margin, dmin=dmin, davg=davg, dmax=dmax, mode=mode, rev=rev)
R = compute(P)
kind, title, msg = R["state"]

# ------------------------------------------------------------------ SALIDA --
st.title("📏 Contracción térmica y dimensional · Tubería PE")
st.caption(f"{material} · α = {alpha:.2f}×10⁻⁴ 1/K · ρ = {rho:.0f} kg/m³ · "
           f"modelo lineal ε = α·ΔT (válido para ΔT moderados, < ~100 K)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Longitud de corte recomendada", f"{R['L_rec'] / 1000:.3f} m",
          f"{R['L_rec'] - R['L_f']:+.0f} mm vs. final", delta_color="off")
c2.metric("Contracción longitudinal", f"{R['shrink_mm']:.1f} mm",
          f"{R['shrink_pct']:.3f} %", delta_color="off")
c3.metric("Ø exterior final", f"{R['d_f']:.2f} mm", f"{R['d_f'] - d0:+.2f} mm", delta_color="off")
c4.metric("Estado vs. norma", {"ok": "✅ En norma", "warn": "⚠️ Al límite", "bad": "⛔ Fuera"}[kind])

st.markdown(f'<div class="status {kind}"><h4>{title}</h4><p>{msg}</p></div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("T media de pared", f"{R['tm']:.1f} °C")
m2.metric("ΔT de contracción", f"{R['dT']:.1f} K")
m3.metric("Espesor final", f"{R['e_f']:.2f} mm", f"SDR {R['sdr']:.1f}", delta_color="off")
m4.metric("Ø interior final", f"{R['di_f']:.2f} mm")
m5.metric("Masa por metro", f"{R['mass_pm']:.2f} kg/m", f"{R['mass_tot']:.1f} kg total", delta_color="off")

if R["dT"] <= 0:
    st.warning("La temperatura final es mayor o igual a la media de pared: la tubería **se dilatará**, no se contraerá.")

tab_len, tab_dia, tab_sens, tab_calc = st.tabs(
    ["📐 Longitud de corte", "⭕ Diámetro vs. norma", "📈 Sensibilidad", "🧮 Memoria de cálculo"])

# ---- Longitud ----------------------------------------------------------------
with tab_len:
    a, b = st.columns(2)
    with a:
        st.subheader("Cálculo inverso")
        st.dataframe(pd.DataFrame({
            "Concepto": ["Metraje final objetivo", "Longitud exacta de corte (sin margen)",
                         "Margen de seguridad", "Longitud de corte recomendada",
                         "Contracción esperada", "Contracción por metro final"],
            "Valor": [f"{R['L_f']:.0f} mm", f"{R['L_exact']:.1f} mm", f"+{margin:.0f} mm",
                      f"{R['L_rec']:.1f} mm  ({R['L_rec'] / 1000:.3f} m)",
                      f"{R['shrink_mm']:.1f} mm ({R['shrink_pct']:.3f} %)",
                      f"{R['shrink_mm'] / L_final:.2f} mm/m"]}), hide_index=True, **STRETCH)
    with b:
        st.subheader("Verificación directa")
        cut = st.number_input("Longitud de corte real (mm)", 100.0, 1_000_000.0,
                              float(round(R["L_rec"], 0)), 1.0, format="%.1f")
        L_res = cut * R["k_L"]
        st.metric("Longitud final resultante", f"{L_res / 1000:.3f} m", f"{L_res - R['L_f']:+.1f} mm vs. objetivo",
                  delta_color="normal")
        if L_res >= R["L_f"]:
            st.success("La tubería NO quedará más corta que el metraje final esperado.")
        else:
            st.error(f"Quedará {R['L_f'] - L_res:.1f} mm más corta que el objetivo.")

# ---- Diámetro ----------------------------------------------------------------
with tab_dia:
    k = R["k_th"]
    st.markdown(f"**Ventana de calibración:** para cumplir la norma al enfriar a {t_final:.0f} °C, el Ø inicial "
                f"(caliente) debe estar entre **{dmin / k:.2f}** y **{dmax / k:.2f} mm**; objetivo "
                f"**{davg / k:.2f} mm**. Tu Ø inicial: **{d0:.2f} mm** "
                f"({'✅ dentro' if dmin / k <= d0 <= dmax / k else '⛔ fuera'} de la ventana).")
    st.dataframe(pd.DataFrame({
        "Límite": ["Mínimo", "Promedio", "Máximo"],
        "Ø norma (mm)": [dmin, davg, dmax],
        "Ø inicial requerido (mm)": [round(v / k, 3) for v in (dmin, davg, dmax)],
        "Ø final predicho (mm)": [round(R["d_f"], 3)] * 3,
        "Margen (mm)": [round(R["d_f"] - dmin, 3), round(R["d_f"] - davg, 3), round(dmax - R["d_f"], 3)],
    }), hide_index=True, **STRETCH)
    st.caption("Margen: positivo = dentro del límite; para «Promedio» es el desvío con signo.")

    tm_ = R["tm"]
    ts = np.linspace(min(t_final, tm_) - 20, max(t_final, tm_) + 5, 80)
    dfr = pd.DataFrame({"T": ts, "D": d0 * (1 - R["a"] * (tm_ - ts))})
    ymin, ymax = min(dfr.D.min(), dmin) - 0.15, max(dfr.D.max(), dmax) + 0.15
    band = pd.DataFrame({"x1": [ts.min()], "x2": [ts.max()], "y1": [dmin], "y2": [dmax]})
    rules = pd.DataFrame({"y": [dmin, davg, dmax], "Límite": ["Mínimo", "Promedio", "Máximo"]})
    x = alt.X("T:Q", title="Temperatura final (°C)", scale=alt.Scale(zero=False))
    ch = (alt.Chart(band).mark_rect(opacity=0.12, color="#2e7d32").encode(
              x="x1:Q", x2="x2:Q", y="y1:Q", y2="y2:Q")
          + alt.Chart(rules).mark_rule(strokeDash=[6, 4], strokeWidth=2).encode(
              y=alt.Y("y:Q", scale=alt.Scale(domain=[ymin, ymax]), title="Ø exterior (mm)"),
              color=alt.Color("Límite:N", scale=alt.Scale(
                  domain=["Mínimo", "Promedio", "Máximo"], range=["#d32f2f", "#2e7d32", "#d32f2f"]),
                  legend=alt.Legend(orient="bottom", title=None)))
          + alt.Chart(dfr).mark_line(strokeWidth=3, color="#1976d2").encode(x=x, y="D:Q")
          + alt.Chart(pd.DataFrame({"T": [t_final], "D": [R["d_f"]]})).mark_point(
              size=160, filled=True, color="#ed6c02").encode(x="T:Q", y="D:Q")
          ).properties(height=340, title="Ø exterior final vs. temperatura final (punto = escenario actual)")
    st.altair_chart(ch, **STRETCH)

# ---- Sensibilidad ------------------------------------------------------------
with tab_sens:
    st.subheader("Escenarios de temperatura final")
    rows = []
    for dt in (-15, -10, -5, 0, 5, 10, 15):
        r = compute({**P, "t_final": t_final + dt})
        rows.append({"T final (°C)": t_final + dt, "Contracción (%)": round(r["shrink_pct"], 3),
                     "Corte recomendado (m)": round(r["L_rec"] / 1000, 3),
                     "Ø final (mm)": round(r["d_f"], 2), "Estado": {"ok": "✅", "warn": "⚠️", "bad": "⛔"}[r["state"][0]]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)

    st.subheader("Comparativo PE 80 vs. PE 100 (mismas condiciones)")
    rows = []
    for name in ("PE 80", "PE 100"):
        r = compute(P, alpha=MATERIALS[name]["alpha"])
        rows.append({"Material": name, "α (×10⁻⁴ 1/K)": MATERIALS[name]["alpha"],
                     "Contracción (%)": round(r["shrink_pct"], 3),
                     "Corte recomendado (m)": round(r["L_rec"] / 1000, 3),
                     "Ø final (mm)": round(r["d_f"], 2), "Estado": {"ok": "✅", "warn": "⚠️", "bad": "⛔"}[r["state"][0]]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)

    lt = pd.DataFrame({"T": ts, "L": (L_final * 1000 / ((1 - R["a"] * (tm_ - ts)) * (1 - rev / 100)) + margin) / 1000})
    lch = (alt.Chart(lt).mark_line(strokeWidth=3, color="#1976d2").encode(
               x=alt.X("T:Q", title="Temperatura final (°C)", scale=alt.Scale(zero=False)),
               y=alt.Y("L:Q", title="Longitud de corte (m)", scale=alt.Scale(zero=False)))
           + alt.Chart(pd.DataFrame({"T": [t_final], "L": [R["L_rec"] / 1000]})).mark_point(
               size=160, filled=True, color="#ed6c02").encode(x="T:Q", y="L:Q")
           ).properties(height=300, title="Longitud de corte requerida vs. temperatura final")
    st.altair_chart(lch, **STRETCH)

# ---- Memoria de cálculo --------------------------------------------------------
with tab_calc:
    st.latex(r"T_m=T_i+(T_e-T_i)\,f \quad;\quad \Delta T=T_m-T_f")
    st.latex(r"k_{th}=1-\alpha\,\Delta T \quad;\quad k_L=k_{th}\,(1-r) \quad;\quad L_0=\dfrac{L_f}{k_L}+m")
    st.latex(r"D_f=D_0\,k_{th} \quad;\quad D_{0,req}=\dfrac{D_{norma}}{k_{th}}")
    st.markdown(f"""
| Variable | Valor |
|---|---|
| T media de pared T_m | {R['tm']:.2f} °C ({mode.lower()}) |
| ΔT = T_m − T_f | {R['dT']:.2f} K |
| α | {alpha:.2f}×10⁻⁴ 1/K |
| k_th | {R['k_th']:.6f} |
| Retracción residual r | {rev:.2f} % |
| k_L | {R['k_L']:.6f} |
| L_f (objetivo) | {R['L_f']:.0f} mm |
| L_0 exacta | {R['L_exact']:.2f} mm |
| Margen m | {margin:.0f} mm |
| **L_0 recomendada** | **{R['L_rec']:.2f} mm** |
| Ø final | {R['d_f']:.3f} mm |
""")
    st.caption("Supuestos: α constante, contracción isotrópica, densidad de referencia a la temperatura final, "
               "T media de pared uniforme para el cálculo del diámetro y la longitud.")

    report = f"""REPORTE DE CONTRACCIÓN - TUBERÍA {material}
Fecha: {datetime.now():%Y-%m-%d %H:%M}
Ø inicial {d0:.2f} mm | espesor {e0:.2f} mm | T ext {t_ext:.1f} °C | T int {t_int:.1f} °C | T final {t_final:.1f} °C
α {alpha:.2f}e-4 1/K | densidad {rho:.0f} kg/m3 | norma Ø min/prom/máx {dmin:.2f}/{davg:.2f}/{dmax:.2f} mm
---
T media pared: {R['tm']:.2f} °C | ΔT: {R['dT']:.2f} K
Contracción longitudinal: {R['shrink_mm']:.1f} mm ({R['shrink_pct']:.3f} %)
Metraje final objetivo: {R['L_f']:.0f} mm
Longitud de corte exacta: {R['L_exact']:.1f} mm | recomendada (+{margin:.0f} mm): {R['L_rec']:.1f} mm
Ø final: {R['d_f']:.3f} mm | espesor final: {R['e_f']:.3f} mm | SDR: {R['sdr']:.1f}
Estado: {title}
"""
    st.download_button("⬇️ Descargar reporte (.txt)", report, file_name="reporte_contraccion_pe.txt")
