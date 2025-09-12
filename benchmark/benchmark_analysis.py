import os
import pandas as pd
import plotly.express as px

log_path = "/log/axo.core.decorators"  

df = pd.read_csv(log_path, names=[
    "timestamp", "ms", "log_level", "module", "thread", 
    "event", "scope", "method", "success", "service_time"
])
df = df[df.success == True]

df["service_time"] = pd.to_numeric(df["service_time"], errors="coerce")
df.dropna(subset=["service_time"], inplace=True)

# Estadísticas básicas por método
stats = df.groupby("method")["service_time"].agg(["mean", "median", "std"]).reset_index()
stats.columns = ["Algoritmo", "Media", "Mediana", "Desviación estándar"]
print(stats)

output_dir = os.path.dirname(os.path.abspath(__file__))

fig_box = px.box(
    df,
    x="method",
    y="service_time",
    color="method",
    title="Distribución de tiempos de ejecución por método",
    labels={"method": "Método", "service_time": "Tiempo de ejecución (s)"}
)
boxplot_path = os.path.join(output_dir, "boxplot_tiempo.html")
fig_box.write_html(boxplot_path)
print("Boxplot generado")

stats_melted = stats.melt(
    id_vars="Algoritmo",
    value_vars=["Media", "Mediana", "Desviación estándar"],
    var_name="Estadístico",
    value_name="Tiempo (s)"
)

fig_bar = px.bar(
    stats_melted,
    x="Algoritmo",
    y="Tiempo (s)",
    color="Estadístico",
    barmode="group",
    title="Estadísticos básicos por algoritmo",
    labels={"Algoritmo": "Método"}
)
barplot_path = os.path.join(output_dir, "barras_estadisticas.html")
fig_bar.write_html(barplot_path)
print("Gráfico de barras generado")
