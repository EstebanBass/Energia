import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import folium
from streamlit.components.v1 import html

def generarMenu():
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open("media/Energia.png")
            st.image(image, use_container_width=False)
        with col2:
            st.header("Energias Limpias TalentoTech 2025")
        
        st.page_link('app.py', label="Energias Limpias")
        st.page_link('pages/Demografia.py', label="Demografia")
        st.page_link('pages/Viento.py', label="Viento")
        st.page_link('pages/Temperatura.py', label="Temperatura")


def consumos(df):
    st.markdown('## Informacion Demografica por consumos\n')
    st.write('')
    st.write("Se realizó una validación cuantitativa de los consumos de la red electrica en los departamentos del pais, y esta información se contrasto cons la información demografica entregada por medios de prensa nacionales, regionales y estudios publicados por el gobierno nacional, con el fin de entender cuales eran las zonas con un mayor nivel de complejidad en la consecución de recursos electricos")
    st.write('')
    st.write("Esta es una muestra de la información ustilizada en la base de datos dela información")
    st.write('')
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
    colores = ['#0066cc', '#0080ff', '#3399ff', '#66b3ff', '#99ccff']

    fig, ax = plt.subplots(figsize=(8, 8))  # Ajustar tamaño para mejor visualización

    # Generar el gráfico de torta
    df_departamento_f.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, cmap='viridis', colors = colores)

    # Ajustar el gráfico para mejor presentación
    ax.set_ylabel('')  # Ocultar el label del eje Y
    ax.set_title('Consumo de Energía por Departamento')

    # Mostrar en Streamlit
    st.pyplot(fig)


    st.markdown('### Empresas con incursion en el sector ')

    #df_empresa_agg = df[df['Departamento'].isin(dep)]
    df_empresa_agg = df.groupby(['Empresa'])['valor_consumo'].sum()
    df_empresa_agg = df_empresa_agg.sort_values(ascending=False)
    df_empresa_agg = df_empresa_agg.nlargest(8)

    fig, ax = plt.subplots(figsize=(12, 5))
    df_empresa_agg.plot(kind='bar', ax=ax, color="royalblue", alpha=0.7)

    # Personalización
    ax.set_title('Top 8 de Empresas por Consumo')
    ax.set_xlabel('Empresa')
    ax.set_ylabel('Valor Consumo')
    ax.tick_params(axis='x')

    # Mostrar en Streamlit
    st.pyplot(fig)
    st.write('')
    st.write('Al analizar la información de estas empresas, vimos que la unica que tiene proyectos relevantes de energias limpias es EMP, cuyo enofque esta en la energia solar, asunque principalmente tiene ejecución en el departamento de antioquia')

def viento(df_final):
    estacio2=df_final[df_final['codigoestacion']==15065190]
    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(12, 6))

    # Graficar cada sensor con un color diferente
    sns.lineplot(data=estacio2, x='fecha', y='valorobservado', hue='codigoestacion', 
                style='departamento', markers=True, ax=ax)

    # Configurar etiquetas y título
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor Observado")
    ax.set_title("Velocidad de Sensores por Departamento")
    ax.legend(title="Sensor / Departamento", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Mostrar en Streamlit
    st.pyplot(fig)


    #Grafico estacion en la Guajira 
    estacionf=df_final[df_final['codigoestacion']==15065180]

    fig, ax = plt.subplots(figsize=(12, 6))

    # ---- GRAFICAR DATOS ----
    sns.lineplot(
        data=estacionf, 
        x='fecha', 
        y='valorobservado', 
        hue='codigoestacion', 
        style='departamento', 
        markers=True, 
        ax=ax
    )

    # ---- CONFIGURACIÓN DE GRÁFICO ----
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor Observado")
    ax.set_title("Velocidad de Sensores por Departamento")
    ax.legend(title="Sensor / Departamento", bbox_to_anchor=(1.05, 1), loc='upper left')

    # ---- MOSTRAR EN STREAMLIT ----
    st.pyplot(fig)







    df_final["fecha"] = pd.to_datetime(df_final["fecha"], format="%Y-%m-%d")
    estacionf = df_final[df_final['codigoestacion'] == 15065180]
    estacionf1 = df_final[df_final['codigoestacion'] == 15065190]
    estacionf2 = df_final[df_final['codigoestacion'] == 15075501]
    estacionf3 = df_final[df_final['codigoestacion'] == 15085050]
    estacionf5 = df_final[df_final['codigoestacion'] == 1508500053]
    estacion21 = df_final[df_final['codigoestacion'] == 33035010]

    # Concatenamos los sensores
    sensoresf = pd.concat([estacionf, estacionf1, estacionf2, estacionf3, estacionf5, estacion21])

    # Hallamos la energía generada
    area = 52
    densidad = 1.2
    coeficiente = 5.93
    sensoresf["Energia (KW)"] = ((sensoresf["valorobservado"] ** 3) * area * densidad * coeficiente) / 1000

    # Crear una columna con el año y mes
    sensoresf['año'] = sensoresf['fecha'].dt.year
    sensoresf['mes'] = sensoresf['fecha'].dt.to_period('M')

    # Agrupar por "codigoestacion", "mes" y "departamento"
    sensorf1 = sensoresf.groupby(["codigoestacion", "mes", "año", "departamento"]).agg({
        "Energia (KW)": "sum",   # Sumar la energía por sensor y mes
        "nombreestacion": "first",
        "latitud": "first",
        "longitud": "first",
    }).reset_index()

    # Mostrar el dataframe en Streamlit
    st.write(sensoresf)

    # Crear gráfico por cada sensor
    sensores_unicos = sensorf1["codigoestacion"].unique()

    for sensor in sensores_unicos:
        df_sensor = sensorf1[sensorf1["codigoestacion"] == sensor]

        # Asegurar que por cada (año, departamento) solo haya una fila (suma de energía)
        df_sensor = df_sensor.groupby(["año", "departamento"])["Energia (KW)"].sum().reset_index()

        # Ordenar por año
        df_sensor = df_sensor.sort_values(by="año")

        # Obtener lista de años y departamentos para ese sensor
        años = df_sensor["año"].unique()
        departamentos = df_sensor["departamento"].unique()

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 6))

        # Posiciones de las barras (una por año)
        x = np.arange(len(años))

        # Inicializar base para apilar
        bottom = np.zeros(len(años))

        # Colores para cada departamento
        colores = plt.cm.Set2(np.linspace(0, 1, len(departamentos)))

        for i, depto in enumerate(departamentos):
            datos_depto = df_sensor[df_sensor["departamento"] == depto].set_index("año")["Energia (KW)"]

            # Asegurar que todos los años existan, con energía 0 si no hay datos ese año
            energia_por_año = datos_depto.reindex(años, fill_value=0)

            # Graficar barra apilada
            ax.bar(x, energia_por_año, bottom=bottom, color=colores[i], label=depto)

            # Añadir texto dentro de cada segmento
            for j, energia in enumerate(energia_por_año):
                if energia > 0:
                    ax.text(j, bottom[j] + energia / 2, f'{energia:.1f}', 
                            ha='center', va='center', fontsize=9, color='black', weight='bold')

            # Actualizar "bottom" para la siguiente capa
            bottom += energia_por_año.values

        # Personalizar gráfico
        ax.set_title(f"Energía Generada - Sensor {sensor}")
        ax.set_xticks(x)
        ax.set_xticklabels(años)
        ax.set_xlabel("Año")
        ax.set_ylabel("Energía Generada (KW)")
        ax.legend(title="Departamento", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # Mostrar gráfico en Streamlit
        st.pyplot(fig)

    # Mapa de la ubicación de los sensores
    # Agrupar solo por sensor, departamento y año
    sensorf1 = sensoresf.groupby(["codigoestacion", "departamento", "año"]).agg({
        "Energia (KW)": "sum",
        "latitud": "mean",
        "longitud": "mean"
    }).reset_index()

    # Crear mapa
    mapa = folium.Map(location=[sensoresf["latitud"].mean(), sensoresf["longitud"].mean()], zoom_start=6)

    # Agrupar para iterar
    sensores_unicos = sensorf1.groupby(["codigoestacion", "departamento"])

    # Agregar marcadores al mapa
    for (codigoestacion, departamento), datos in sensores_unicos:
        lat = datos["latitud"].mean()
        lon = datos["longitud"].mean()

        energia_por_año = "<br>".join(
            [f"<b>{año}:</b> {energia:.1f} KW" for año, energia in zip(datos["año"], datos["Energia (KW)"])]
        )

        popup_info = f"""
        <b>Sensor:</b> {codigoestacion} <br>
        <b>Departamento:</b> {departamento} <br>
        <b>Energía Generada por Año:</b><br>{energia_por_año}
        """

        folium.Marker(
            location=[lat, lon],
            popup=popup_info,
            tooltip=f"Sensor {codigoestacion}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(mapa)

    # Mostrar mapa en Streamlit
    st.write("### Mapa de Sensores")
    st.components.v1.html(mapa._repr_html_(), height=500)


def temperatura(df_temperatura):
    st.markdown('## Análisis de Temperatura para los departamentos de La Guajira, Meta y Putumayo.\n')
    st.write("Se toma como fuente de información los datos de temperatura en un rango de tiempo del 01 de Enero del 2020 a el 4 de Febrero del 2025.")
    st.write("Fuente de datos: https://www.datos.gov.co/Ambiente-y-Desarrollo-Sostenible/Temperatura/xq6k-pr9h")
    
    #%% Cambiamos el tipo de dato Fecha Observacion de tipo object a datetime64[ns]
    df_temperatura['FechaObservacion'] = pd.to_datetime(df_temperatura['FechaObservacion'])
    df_temp=df_temperatura
    # solo tomamos los valores de 6am a 6pm
    df_temperatura= df_temperatura[(df_temperatura['FechaObservacion'].dt.time >= pd.to_datetime('06:00:00').time()) & (df_temperatura['FechaObservacion'].dt.time <= pd.to_datetime('18:00:00').time())]
    #%% Separamos los datos por departamento
    df_Meta1=df_temp[df_temp['Departamento'] == 'META']
    df_Guajira1=df_temp[df_temp['Departamento'] == 'LA GUAJIRA']
    df_Putumayo1=df_temp[df_temp['Departamento'] == 'PUTUMAYO']
    arr_Guajira=df_Guajira1['CodigoEstacion'].unique()
    arr_Putumayo=df_Putumayo1['CodigoEstacion'].unique()
    arr_Meta=df_Meta1['CodigoEstacion'].unique()
    
    st.write("Para cada departamento se tienen la siguiente cantidad de estaciones donde se toma la temperatura:")
    #Acá va la tabla:
    # Datos
    data = {
        'Departamento': ['La Guajira','Meta', 'Putumayo'],
        'Cantidad Estaciones': [len(arr_Guajira), len(arr_Meta), len(arr_Putumayo)]
    }

    # Crear DataFrame
    df = pd.DataFrame(data)
    # Mostrar tabla en Streamlit
    st.table(df)
    #%% df Filtrados de 6am a 6pm por departamento
    df_Meta=df_temperatura[df_temperatura['Departamento'] == 'META']
    df_Guajira=df_temperatura[df_temperatura['Departamento'] == 'LA GUAJIRA']
    df_Putumayo=df_temperatura[df_temperatura['Departamento'] == 'PUTUMAYO']
    
    #Se Filtran todos los df para los valores entre 15 y 35°C
    df_15a25P = df_Putumayo[(df_Putumayo['ValorObservado_°C'] >= 15) & (df_Putumayo['ValorObservado_°C'] <= 35)]
    df_15a25G = df_Guajira[(df_Guajira['ValorObservado_°C'] >= 15) & (df_Guajira['ValorObservado_°C'] <= 35)]
    df_15a25M = df_Meta[(df_Meta['ValorObservado_°C'] >= 15) & (df_Meta['ValorObservado_°C'] <= 35)]
    #%% Organizamos los dataframe por orden cronologico
    df_15a25P=df_15a25P.sort_values(by='FechaObservacion').reset_index(drop=True)
    df_15a25G=df_15a25G.sort_values(by='FechaObservacion').reset_index(drop=True)
    df_15a25M=df_15a25M.sort_values(by='FechaObservacion').reset_index(drop=True)
    # Contar la frecuencia de las estaciones en el dataframe filtrado
    f_CodEstacP = df_15a25P['CodigoEstacion'].value_counts()
    f_CodEstacP = f_CodEstacP.sort_values(ascending=False)
    f_CodEstacG = df_15a25G['CodigoEstacion'].value_counts()
    f_CodEstacG = f_CodEstacG.sort_values(ascending=False)
    f_CodEstacM = df_15a25M['CodigoEstacion'].value_counts()
    f_CodEstacM = f_CodEstacM.sort_values(ascending=False)
    dfs_codEstGMP = {
        1: f_CodEstacG,
        2: f_CodEstacM,
        3: f_CodEstacP
    }
    dfs_Dept = {
        1: 'La Guajira',
        2: 'Meta',
        3: 'Putumayo'
    }
    st.write("Tomando como base que la mayor eficiencia energética en paneles solares se da cuando estos reciben temperaturas entre los 15°C y 35°C, se decide filtrar los datos para estas temperaturas entre las 6am y 6pm.")
    st.write("Se generan gráficas de barras por cada Departamento donde se muestran la cantidad de veces por estación en las que se presentaron las temperaturas entre 15°C y 35°C con el fin de seleccionar las 3 mejores estaciones por Departamento.")
    for i in range(1, 4):
        # Crear la figura y el eje
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plotear los datos en el eje
        dfs_codEstGMP[i].plot(kind='bar', ax=ax)
        ax.set_title('Frecuencia de Temperatura en Estaciones de {}'.format(dfs_Dept[i]))
        ax.set_xlabel('Código de Estación')
        ax.set_ylabel('Frecuencia de Valores de Temperatura (15°C-25°C)')
        ax.set_xticks(range(len(dfs_codEstGMP[i])))  # Establecer los valores del eje X
        ax.set_xticklabels(dfs_codEstGMP[i].index, rotation=45)  # Etiquetas del eje X con rotación
        
        # Mostrar la cuadrícula solo en el eje Y
        ax.grid(axis='y')

        # Mostrar la gráfica en Streamlit
        st.pyplot(fig)
    #Se toman las 3 mejores estaciones por departamento:
    codEstGuajira1,codEstGuajira2,codEstGuajira3=f_CodEstacG.index[0],f_CodEstacG.index[1],f_CodEstacG.index[2]
    codEstMeta1,codEstMeta2,codEstMeta3=f_CodEstacM.index[0],f_CodEstacM.index[1],f_CodEstacM.index[2]
    codEstPutumayo1,codEstPutumayo2,codEstPutumayo3=f_CodEstacP.index[0],f_CodEstacP.index[1],f_CodEstacP.index[2]
    #%%Obtenemos todos los valores de estos codigo de estación
    df_EficP1,df_EficP2,df_EficP3= df_15a25P[df_15a25P['CodigoEstacion'] == codEstPutumayo1],df_15a25P[df_15a25P['CodigoEstacion'] == codEstPutumayo2],df_15a25P[df_15a25P['CodigoEstacion'] == codEstPutumayo3]
    df_EficM1,df_EficM2,df_EficM3= df_15a25M[df_15a25M['CodigoEstacion'] == codEstMeta1],df_15a25M[df_15a25M['CodigoEstacion'] == codEstMeta2],df_15a25M[df_15a25M['CodigoEstacion'] == codEstMeta3]
    df_EficG1,df_EficG2,df_EficG3= df_15a25G[df_15a25G['CodigoEstacion'] == codEstGuajira1],df_15a25G[df_15a25G['CodigoEstacion'] == codEstGuajira2],df_15a25G[df_15a25G['CodigoEstacion'] == codEstGuajira3]
    # Creamos una nueva columna para los meses y los años
    df_EficP1['Año'],df_EficP1['Mes'] = df_EficP1['FechaObservacion'].dt.year,df_EficP1['FechaObservacion'].dt.month
    df_EficP2['Año'],df_EficP2['Mes'] = df_EficP2['FechaObservacion'].dt.year,df_EficP2['FechaObservacion'].dt.month
    df_EficP3['Año'],df_EficP3['Mes'] = df_EficP3['FechaObservacion'].dt.year,df_EficP3['FechaObservacion'].dt.month
    df_EficG1['Año'],df_EficG1['Mes'] = df_EficG1['FechaObservacion'].dt.year,df_EficG1['FechaObservacion'].dt.month
    df_EficG2['Año'],df_EficG2['Mes'] = df_EficG2['FechaObservacion'].dt.year,df_EficG2['FechaObservacion'].dt.month
    df_EficG3['Año'],df_EficG3['Mes'] = df_EficG3['FechaObservacion'].dt.year,df_EficG3['FechaObservacion'].dt.month
    df_EficM1['Año'],df_EficM1['Mes'] = df_EficM1['FechaObservacion'].dt.year,df_EficM1['FechaObservacion'].dt.month
    df_EficM2['Año'],df_EficM2['Mes'] = df_EficM2['FechaObservacion'].dt.year,df_EficM2['FechaObservacion'].dt.month
    df_EficM3['Año'],df_EficM3['Mes'] = df_EficM3['FechaObservacion'].dt.year,df_EficM3['FechaObservacion'].dt.month

    # Agrupamos por año y mes, y calcular el valor promedio Putumayo
    df_groupedP1 = df_EficP1.groupby(['Año', 'Mes'])['ValorObservado_°C'].mean().unstack(level=0)
    df_groupedP2 = df_EficP2.groupby(['Año', 'Mes'])['ValorObservado_°C'].mean().unstack(level=0)
    df_groupedP3 = df_EficP3.groupby(['Año', 'Mes'])['ValorObservado_°C'].mean().unstack(level=0)
    dfs_groupedP = {
        1: df_groupedP1,
        2: df_groupedP2,
        3: df_groupedP3
    }
    dfs_eficP = {
        1: df_EficP1,
        2: df_EficP2,
        3: df_EficP3
    }
    # Agrupamos por año y mes, y calcular el valor promedio Guajira
    df_groupedG1 = df_EficP1.groupby(['Año', 'Mes'])['ValorObservado_°C'].mean().unstack(level=0)
    df_groupedG2 = df_EficP2.groupby(['Año', 'Mes'])['ValorObservado_°C'].mean().unstack(level=0)
    df_groupedG3 = df_EficP3.groupby(['Año', 'Mes'])['ValorObservado_°C'].mean().unstack(level=0)
    dfs_groupedG = {
        1: df_groupedG1,
        2: df_groupedG2,
        3: df_groupedG3
    }
    dfs_eficG = {
        1: df_EficG1,
        2: df_EficG2,
        3: df_EficG3
    }
    # Agrupamos por año y mes, y calcular el valor promedio de Meta
    df_groupedM1 = df_EficP1.groupby(['Año', 'Mes'])['ValorObservado_°C'].mean().unstack(level=0)
    df_groupedM2 = df_EficP2.groupby(['Año', 'Mes'])['ValorObservado_°C'].mean().unstack(level=0)
    df_groupedM3 = df_EficP3.groupby(['Año', 'Mes'])['ValorObservado_°C'].mean().unstack(level=0)
    dfs_groupedM = {
        1: df_groupedM1,
        2: df_groupedM2,
        3: df_groupedM3
    }
    dfs_eficM = {
        1: df_EficM1,
        2: df_EficM2,
        3: df_EficM3
    }
    st.write("Se grafican los valores de temperatura promedio por mes en cada año para las 3 mejores estaciones de La Guajira.")
    for i in range(1, 4):
    # Creamos la figura y el eje
        fig, ax = plt.subplots(figsize=(12, 6))

        # Ploteamos los datos en el eje
        dfs_groupedG[i].plot(kind='line', marker='o', ax=ax)
        ax.set_title('Valor promedio de Temperatura en {} {} codEstacion: {}'.format(dfs_eficG[i]['Municipio'].iloc[0], dfs_eficG[i]['Departamento'].iloc[0], dfs_eficG[i]['CodigoEstacion'].iloc[0]))
        ax.set_xlabel('Mes')
        ax.set_ylabel('Valor Observado (°C)')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(range(1, 13))  # Establecer los valores del eje X de 1 a 12
        ax.legend(title='Año')
        ax.grid(True)
        
        # Mostrar la gráfica en Streamlit
        st.pyplot(fig)
    st.write("Se grafican los valores de temperatura promedio por mes en cada año para las 3 mejores estaciones de Meta.")
    for i in range(1, 4):
    # Creamos la figura y el eje
        fig, ax = plt.subplots(figsize=(12, 6))

        # Ploteamos los datos en el eje
        dfs_groupedM[i].plot(kind='line', marker='o', ax=ax)
        ax.set_title('Valor promedio de Temperatura en {} {} codEstacion: {}'.format(dfs_eficM[i]['Municipio'].iloc[0], dfs_eficM[i]['Departamento'].iloc[0], dfs_eficM[i]['CodigoEstacion'].iloc[0]))
        ax.set_xlabel('Mes')
        ax.set_ylabel('Valor Observado (°C)')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(range(1, 13))  # Establecer los valores del eje X de 1 a 12
        ax.legend(title='Año')
        ax.grid(True)
        
        # Mostrar la gráfica en Streamlit
        st.pyplot(fig)
    st.write("Se grafican los valores de temperatura promedio por mes en cada año para las 3 mejores estaciones de Putumayo.")
    for i in range(1, 4):
    # Creamos la figura y el eje
        fig, ax = plt.subplots(figsize=(12, 6))

        # Ploteamos los datos en el eje
        dfs_groupedP[i].plot(kind='line', marker='o', ax=ax)
        ax.set_title('Valor promedio de Temperatura en {} {} codEstacion: {}'.format(dfs_eficP[i]['Municipio'].iloc[0], dfs_eficP[i]['Departamento'].iloc[0], dfs_eficP[i]['CodigoEstacion'].iloc[0]))
        ax.set_xlabel('Mes')
        ax.set_ylabel('Valor Observado (°C)')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(range(1, 13))  # Establecer los valores del eje X de 1 a 12
        ax.legend(title='Año')
        ax.grid(True)
        
        # Mostrar la gráfica en Streamlit
        st.pyplot(fig)
   