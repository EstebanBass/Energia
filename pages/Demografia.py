import pandas as pd
import utilidades as util
import streamlit as st

util.generarMenu()

df = pd.read_csv("data/info_unificado.csv", index_col=0)


util.consumos(df)