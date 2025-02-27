import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def generarMenu():
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open("media/imagen_2.jpeg")
            st.image(image, use_container_width=False)
        with col2:
            st.header("SMEC")
        
        st.page_link('app.py', label="Inicio")
        st.page_link('pages/Demografia.py', label="Demografia")
        st.page_link('pages/Demografia.py', label="Viento")
        st.page_link('pages/Demografia.py', label="Temperatura")


def consumos(df):
    st.markdown('## Informacion Demografica por consumos')
    st.write(df.head(2))
    st.subheader('Consumo por grupo de personas')
    
    dep = ['META', 'PUTUMAYO', 'LA GUAJIRA']

    df_departamento = df.groupby(['Departamento'])['valor_consumo'].sum()
    df_departamento = df_departamento.sort_values(ascending=False)
    df_departamento.plot(kind='bar', figsize=(12,5), title='Consumo de Energia por departamento')


    df_departamento_f = df_departamento[df_departamento.index.isin(dep)]
    df_departamento_f.plot(kind='bar', figsize=(12,5), title='Consumo de Energia por departamento')


    st.markdown('### Separamos los datos')
   