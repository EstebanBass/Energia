import pandas as pd
import matplotlib.pyplot as plt
import utilidades as util
import streamlit as st
import seaborn as sns

util.generarMenu()

df_temperatura = pd.read_csv("data/Temperatura_20250221_Putumayo_Meta_Guajira_Final.csv")

