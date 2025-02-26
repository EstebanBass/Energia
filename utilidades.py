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
            image = Image.open("media\icono_pag2.png")
            st.image(image, use_container_width=False)
        with col2:
            st.header("SMEC")
        
        st.page_link('app.py', label="Inicio")
        st.page_link('pages/pronostico.py', label='Pronostico')
    
#Funcion del modelo predictivo
'''

def modelo_rf(df_p):
    st.markdown('## Datos Enfermedades de Pacientes')
    st.write(df_p.head())
    st.subheader('Resultado del modelo Ramdom Forest')
    #Variable a predecir
    y = df_p.iloc[:,0]
    #Variables predictoras
    x = df_p.iloc[:,1:]
    #Variables de prueba -> prueba
    #Variables de entrenamiento -> entrenar
    x_entrenar, x_prueba, y_entrenar, y_prueba = train_test_split(x, y,train_size=0.8, random_state=42)

    st.markdown('### Separamos los datos')
    st.write('Datos de entrenamiento')
    st.info(f'Muestra de variables de entrenamiento: {x_entrenar.shape[0]} datos')
    st.info(f'Muestra de variables de prueba: {x_entrenar.shape[1]} datos')

    #CReamos el bosque

    bosque = RandomForestClassifier()

    #Entrenar el bosque

    bosque.fit(x_entrenar, y_entrenar)

    #Hacemos la prediccion

    y_prediccion = bosque.predict(x_prueba)
    accuracy = accuracy_score(y_prueba, y_prediccion)
    st.write('Metrica de precisión de puntos obtenidos')
    st.info(accuracy)'''