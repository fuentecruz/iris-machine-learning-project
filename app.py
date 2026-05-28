import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------

st.set_page_config(
    page_title="Clasificación Iris",
    layout="wide"
)

st.title("Dashboard de Clasificación de Flores Iris")

st.write("""
Este dashboard permite realizar predicciones de especies
de flores Iris utilizando un modelo Random Forest entrenado
previamente.
""")

# ---------------------------------
# CARGAR MODELO Y DATASET
# ---------------------------------

model = joblib.load('iris_model.pkl')

df = pd.read_csv('iris.csv')

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.header("Ingrese las características")

sepal_length = st.sidebar.slider(
    "Sepal Length",
    float(df['SepalLengthCm'].min()),
    float(df['SepalLengthCm'].max()),
    5.0
)

sepal_width = st.sidebar.slider(
    "Sepal Width",
    float(df['SepalWidthCm'].min()),
    float(df['SepalWidthCm'].max()),
    3.0
)

petal_length = st.sidebar.slider(
    "Petal Length",
    float(df['PetalLengthCm'].min()),
    float(df['PetalLengthCm'].max()),
    4.0
)

petal_width = st.sidebar.slider(
    "Petal Width",
    float(df['PetalWidthCm'].min()),
    float(df['PetalWidthCm'].max()),
    1.0
)

# ---------------------------------
# PREDICCIÓN
# ---------------------------------

input_data = pd.DataFrame({
    'SepalLengthCm': [sepal_length],
    'SepalWidthCm': [sepal_width],
    'PetalLengthCm': [petal_length],
    'PetalWidthCm': [petal_width]
})

prediction = model.predict(input_data)

st.subheader("Predicción del Modelo")

st.success(f"La especie predicha es: {prediction[0]}")

# ---------------------------------
# MÉTRICAS
# ---------------------------------

st.subheader("Métricas del Dataset")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Registros",
    len(df)
)

col2.metric(
    "Cantidad de Variables",
    df.shape[1]
)

col3.metric(
    "Especies",
    df['Species'].nunique()
)

# ---------------------------------
# GRÁFICO 3D
# ---------------------------------

st.subheader("Visualización 3D")

fig = px.scatter_3d(
    df,
    x='SepalLengthCm',
    y='SepalWidthCm',
    z='PetalLengthCm',
    color='Species',
    size='PetalWidthCm',
    opacity=0.8
)

fig.update_layout(
    width=900,
    height=600
)

st.plotly_chart(fig)

# ---------------------------------
# TABLA DE DATOS
# ---------------------------------

st.subheader("Dataset Iris")

st.dataframe(df)