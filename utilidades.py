import pandas as pd
import matplotlib.pyplot as plt
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
            st.header("Energias Limpias TalentoTech 2025")
        
        st.page_link('app.py', label="Energias Limpias")
        st.page_link('pages/Demografia.py', label="Demografia")
        st.page_link('pages/Demografia.py', label="Viento")
        st.page_link('pages/Demografia.py', label="Temperatura")


def consumos(df):
    st.markdown('## Informacion Demografica por consumos\n')
    st.write("Se realizó una validación cuantitativa de los consumos de la red electrica en los departamentos del pais, y esta información se contrasto cons la información demografica entregada por medios de prensa nacionales, regionales y estudios publicados por el gobierno nacional, con el fin de entender cuales eran las zonas con un mayor nivel de complejidad en la consecución de recursos electricos")
    st.write(df.head(2))
    st.subheader('Consumo por grupo de personas')
    
    dep = ['META', 'PUTUMAYO', 'LA GUAJIRA']

    df_departamento = df.groupby(['Departamento'])['valor_consumo'].sum()
    df_departamento = df_departamento.sort_values(ascending=False)
    df_departamento.plot(kind='bar', figsize=(12,5), title='Consumo de Energia por departamento')

    fig, ax = plt.subplots(figsize=(12, 5))

    # Generar el gráfico dentro de la figura
    df_departamento.plot(kind='bar', ax=ax, title='Consumo de Energía por Departamento')

    # Mostrar en Streamlit
    st.pyplot(fig)


    df_departamento_f = df_departamento[df_departamento.index.isin(dep)]
    df_departamento_f.plot(kind='bar', figsize=(12,5), title='Consumo de Energia por departamento')
    colores = ['#004d00', '#007f0e', '#33cc33', '#99ff99', '#ccffcc']

    fig, ax = plt.subplots(figsize=(8, 8))  # Ajustar tamaño para mejor visualización

    # Generar el gráfico de torta
    df_departamento_f.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, cmap='viridis', colors = colores)

    # Ajustar el gráfico para mejor presentación
    ax.set_ylabel('')  # Ocultar el label del eje Y
    ax.set_title('Consumo de Energía por Departamento')

    # Mostrar en Streamlit
    st.pyplot(fig)


    st.markdown('### Separamos los datos')
   