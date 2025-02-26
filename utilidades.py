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
        st.page_link('pages/pronostico.py', label='Pronostico')
    
