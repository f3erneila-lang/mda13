import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="FitLife | EDA Económico", layout="wide")

# -----------------------
# Helpers
# -----------------------
@st.cache_data
def load_data(members_path: str, context_path: str):
    context = pd.read_csv(r"C:\Users\ferna\Desktop\Apuntes Clase\Ejercicios python caseros\fitlife_context.csv")

    members = pd.read_csv(r"C:\Users\ferna\Desktop\Apuntes Clase\Ejercicios python caseros\fitlife_members.csv")

    # Normalizar tipos
    members["month"] = pd.to_datetime(members["month"], format="%Y-%m", errors="coerce")
    context["month"] = pd.to_datetime(context["month"], format="%Y-%m", errors="coerce")

    # Merge (relación por month)
    df = pd.merge(members, context, on="month", how="left")

    # Features económicas básicas
    df["is_churned"] = (df["status"] == "churned").astype(int)
    df["margin"] = df["price_paid"] - df["cost_to_serve"]              # margen de contribución mensual
    df["competitor_gap"] = df["price_paid"] - df["competitor_lowcost_price"]  # brecha vs competidor

    # Limpieza de categoricas con muchos NaN
    for col in ["campaign_active", "service_incident", "churn_reason"]:
        if col in df.columns:
            df[col] = df[col].fillna("none")

    return df

def line_plot(x, ys_dict, title, ylabel):
    fig, ax = plt.subplots()
    for label, y in ys_dict.items():
        ax.plot(x, y, marker="o", linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Mes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

def bar_plot(series, title, ylabel):
    fig, ax = plt.subplots()
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig)

# -----------------------
# UI
# -----------------------
st.title("FitLife | EDA con enfoque económico")
st.caption("Churn, márgenes, elasticidad (competidor), LTV vs CAC (proxy), cohorts y saturación operativa.")

with st.sidebar:
    st.header("Carga de datos")
    st.write("Opción A: escribe rutas locales (Windows).")
    members_path = st.text_input(
        "Ruta fitlife_members.csv",
        value=r"C:\Users\ferna\Desktop\Apuntes Clase\Ejercicios python caseros\fitlife_members.csv"
    )
    context_path = st.text_input(
        "Ruta fitlife_context.csv",
        value=r"C:\Users\ferna\Desktop\Apuntes Clase\Ejercicios python caseros\fitlife_context.csv"
    )

    st.divider()
    st.header("Filtros")
    plan_filter = st.multiselect("Plan", ["basic", "premium", "family"], default=["basic", "premium", "family"])
    center_filter = st.multiselect(
        "Centro", ["downtown", "northside", "eastpark", "westfield", "southgate"],
        default=["downtown", "northside", "eastpark", "westfield", "southgate"]
    )

# Load
df = load_data(members_path, context_path)

# Apply filters
df_f = df[df["plan"].isin(plan_filter) & df["center"].isin(center_filter)].copy()

# Date filters
min_m = df_f["month"].min()
max_m = df_f["month"].max()
colA, colB = st.columns(2)
with colA:
    start = st.date_input("Inicio", value=min_m.date())
with colB:
    end = st.date_input("Fin", value=max_m.date())

df_f = df_f[(df_f["month"] >= pd.to_datetime(start)) & (df_f["month"] <= pd.to_datetime(end))]

# -----------------------
# KPI strip
# -----------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Observaciones (socio-mes)", f"{len(df_f):,}".replace(",", "."))
with k2:
    churn_rate = df_f["is_churned"].mean()
    st.metric("Churn rate (global)", f"{churn_rate:.1%}")
with k3:
    avg_margin = df_f["margin"].mean()
    st.metric("Margen medio mensual", f"{avg_margin:.2f} €")
with k4:
    rev = df_f["price_paid"].sum()
    st.metric("Revenue total (periodo)", f"{rev:,.0f} €".replace(",", "."))

st.divider()

# -----------------------
# 1) Composición por plan (estructura de ingresos)
# -----------------------
st.subheader("1) Estructura de ingresos y márgenes (visión economista)")
c1, c2, c3 = st.columns(3)

with c1:
    share = df_f["plan"].value_counts(normalize=True).sort_index()
    bar_plot(share, "Mix de planes (share de socio-mes)", "Proporción")

with c2:
    rev_by_plan = df_f.groupby("plan")["price_paid"].sum().sort_values(ascending=False)
    bar_plot(rev_by_plan, "Revenue total por plan (€)", "€")

with c3:
    margin_by_plan = df_f.groupby("plan")["margin"].mean().sort_values(ascending=False)
    bar_plot(margin_by_plan, "Margen medio mensual por plan (€)", "€")

st.caption("Lectura económica: si basic tiene churn alto y margen bajo, bajar precio puede destruir el margen de contribución.")

st.divider()

# -----------------------
# 2) Churn por plan a lo largo del tiempo
# -----------------------
st.subheader("2) Churn en el tiempo por plan (dinámica de retención)")
tmp = (
    df_f.groupby(["month", "plan"])["is_churned"]
    .mean()
    .reset_index()
)

pivot = tmp.pivot(index="month", columns="plan", values="is_churned").fillna(0).sort_index()
line_plot(
    pivot.index,
    {col: pivot[col] for col in pivot.columns},
    "Churn rate mensual por plan",
    "Churn rate"
)

st.caption("Si el churn del basic sube mientras premium/family se mantienen, suele ser un problema de segmento (sensibilidad precio/valor), no del producto global.")

st.divider()

# -----------------------
# 3) Competidor y churn del basic (elasticidad / sustitución)
# -----------------------
st.subheader("3) Competidor low-cost vs churn del plan basic (señal de elasticidad)")
basic = df_f[df_f["plan"] == "basic"].copy()

if len(basic) > 0:
    basic_month = basic.groupby("month").agg(
        churn_rate=("is_churned", "mean"),
        competitor_price=("competitor_lowcost_price", "mean"),
        gap=("competitor_gap", "mean"),
    ).sort_index()

    fig, ax1 = plt.subplots()
    ax1.plot(basic_month.index, basic_month["churn_rate"], marker="o", linewidth=1.8)
    ax1.set_ylabel("Churn rate basic")
    ax1.set_xlabel("Mes")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(basic_month.index, basic_month["competitor_price"], marker="s", linewidth=1.8)
    ax2.set_ylabel("Precio competidor (€)")

    ax1.set_title("Basic: churn vs precio competidor (doble eje)")
    st.pyplot(fig)

    fig2, ax = plt.subplots()
    ax.scatter(basic["competitor_lowcost_price"], basic["is_churned"], alpha=0.2)
    ax.set_title("Relación (simple): churn individual vs precio competidor")
    ax.set_xlabel("Precio competidor (€)")
    ax.set_ylabel("Churn (0/1)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig2)

    st.caption("Lectura económica: si el churn sube cuando el competidor baja precio, hay sustitución. Si no, el driver suele ser engagement/valor.")
else:
    st.info("No hay datos de plan basic en el filtro actual.")

st.divider()

# -----------------------
# 4) LTV vs CAC (proxy) usando margen y tenure
# -----------------------
st.subheader("4) LTV vs CAC (proxy) por plan (economía unitaria)")
# Proxy de LTV: margen mensual * tenure_months (muy simplificado)
df_f["ltv_proxy"] = df_f["margin"] * df_f["tenure_months"]
ltv_plan = df_f.groupby("plan")["ltv_proxy"].median().sort_values(ascending=False)
cac_med = df_f["acquisition_cost_avg"].median()

c1, c2 = st.columns(2)
with c1:
    bar_plot(ltv_plan, "LTV proxy mediano por plan (margen*tenure)", "€ (proxy)")
with c2:
    st.metric("CAC mediano (contexto)", f"{cac_med:.2f} €")
    st.write("Regla típica: LTV >> CAC para que el modelo sea sostenible.")
    st.write("Ojo: es un proxy (no es LTV real por cohortes), pero orienta la economía unitaria.")

st.divider()

# -----------------------
# 5) Canales de adquisición: calidad del cliente
# -----------------------
st.subheader("5) Canales de adquisición (calidad: tenure, churn, margen)")
channel = df_f.groupby("acquisition_channel").agg(
    churn_rate=("is_churned", "mean"),
    avg_tenure=("tenure_months", "mean"),
    avg_margin=("margin", "mean"),
    n=("member_id", "count")
).sort_values("n", ascending=False)

st.dataframe(channel)

fig, ax = plt.subplots()
ax.bar(channel.index, channel["churn_rate"])
ax.set_title("Churn rate por canal")
ax.set_ylabel("Churn rate")
ax.set_xticklabels(channel.index, rotation=30, ha="right")
ax.grid(True, axis="y", alpha=0.3)
st.pyplot(fig)

st.caption("Lectura económica: un canal puede traer volumen pero mala calidad (tenure bajo, churn alto). Eso destruye LTV/CAC.")

st.divider()

# -----------------------
# 6) Saturación operativa: ocupación vs churn
# -----------------------
st.subheader("6) Saturación operativa (ocupación) y churn")
occ = df_f.groupby("month").agg(
    churn=("is_churned", "mean"),
    occ=("avg_occupancy_rate", "mean")
).sort_index()

fig, ax1 = plt.subplots()
ax1.plot(occ.index, occ["churn"], marker="o", linewidth=1.8)
ax1.set_ylabel("Churn rate")
ax1.set_xlabel("Mes")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(occ.index, occ["occ"], marker="s", linewidth=1.8)
ax2.set_ylabel("Ocupación media_")