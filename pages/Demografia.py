import pandas as pd
import matplotlib.pyplot as plt
import utilidades as util
import streamlit as st

util.generarMenu()

df = pd.read_csv("data/info_unificado.csv", index_col=0)


util.consumos(df)