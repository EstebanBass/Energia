import pandas as pd
import utilidades as util
import streamlit as st

util.generarMenu()

st.title("Sindrome Metabolico de Enfernmedad Cardiovascular")
df = pd.read_csv("data/Datos_Pacientes.csv", index_col=0)


util.modelo_rf(df)