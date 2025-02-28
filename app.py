import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import utilidades as util
from PIL import Image
import seaborn as sns
import plotly.express as px
import numpy as np

util.generarMenu()

st.header('Proyecto de evaluación de energias renovables TalentoTech')

st.write('En Colombia, el acceso a la energía sigue siendo un desafío en muchas zonas rurales, afectando la calidad de vida y el desarrollo económico de las comunidades. Este proyecto surge de la necesidad de explorar soluciones sostenibles para reducir la brecha energética en regiones con acceso limitado al servicio eléctrico.') 
campo = Image.open("media/campo.png")
st.image(campo, use_container_width=False)        
st.write('')
st.write('A nivel global, la transición hacia fuentes de energía renovable ha permitido sustituir progresivamente los sistemas tradicionales por opciones más sostenibles. En el país, el caso de La Guajira representa un referente en la implementación de energías renovables, donde la generación hidroeléctrica ha sido complementada y, en algunos casos, sustituida por fuentes eólicas y solares.')
st.write('Para el desarrollo de este estudio, se utilizaron datos gubernamentales sobre la disponibilidad de energía en los departamentos seleccionados, así como información satelital que permite evaluar posibles ubicaciones para la instalación de plantas de generación renovable. El objetivo es proporcionar un análisis preliminar de la capacidad energética potencial en Putumayo y Meta, estableciendo una base para futuros estudios más detallados.')