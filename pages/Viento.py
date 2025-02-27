import pandas as pd
import matplotlib.pyplot as plt
import utilidades as util
import streamlit as st
import seaborn as sns

util.generarMenu()

df_final=pd.read_csv("data/df_final1Agrupados.csv")

util.viento(df_final)