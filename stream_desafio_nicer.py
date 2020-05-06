########################################## LIBRERIAS ###########################################

import numpy as np ; import pandas as pd ; import geopandas as gpd
import streamlit as st ; import pickle
import seaborn as sns ; 

import matplotlib.pyplot as plt
import seaborn as sns                     ;   import shapely.wkt

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

########################################## LIBRERIAS ###########################################
#-----------------------------------------------------------------------------------------------
########################################## MAPA BASE ###########################################

crs = {'init': 'epsg:4326'}
barrios = pd.read_csv('carto/barrios_capital.csv',sep=',',encoding='Latin1')

def from_wkt(df, wkt_column):
    # con wkt genero una columna
    df['geometry'] = df[wkt_column].apply(shapely.wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    return gdf

barriosmap = from_wkt(barrios, "WKT")
barriosmap['BARRIO'] = [barriosmap.BARRIO.iloc[i].title() for i in range(len(barriosmap))]

barriosmap.crs = crs
barriosmap.loc[barriosmap.BARRIO == 'Constitucion','BARRIO'] = 'Constitución'
barriosmap.loc[barriosmap.BARRIO == 'Villa Gral. Mitre','BARRIO'] = 'Villa General Mitre'
barriosmap.loc[barriosmap.BARRIO == 'Agronomia','BARRIO'] = 'Agronomía'
barriosmap.loc[barriosmap.BARRIO == 'Nueva Pompeya','BARRIO'] = 'Pompeya'
barriosmap.loc[barriosmap.BARRIO == 'Villa Pueyrredon','BARRIO'] = 'Villa Pueyrredón'
barriosmap.loc[barriosmap.BARRIO == 'San Nicolas','BARRIO'] = 'San Nicolás'


lugares = list(barriosmap.BARRIO.unique())
lugares.sort()

########################################## MAPA BASE ###########################################
#-----------------------------------------------------------------------------------------------
######################################## OBJETOS PICKLE ########################################


with open('objetos_pk.pkl', 'rb') as ele:
    elementos = pickle.load(ele)


model2 = elementos[0]
features_needed = elementos[1]
lista_betas = elementos[2]
stdscaler_X = elementos[3]
stdscaler_y = elementos[4]


######################################## OBJETOS PICKLE ########################################
#-----------------------------------------------------------------------------------------------
###################################### FUNCION PREDICTORA ######################################
def name(**variables):
    ''' esta funcion nos devuelve el nombre de una variable en string,
        para poder concatenar con su valor'''
    return [x for x in variables][0]

def datos_nuevos(loc_):
    global new_data,prop_,size_,superficie_m2,ambientes,amenities,prediccion
    new_data = []

    for i in list(features_needed):
        new_prop = name(prop_=prop_)+prop_
        new_size = name(size_=size_)+size_
        new_loc = name(loc_=loc_)+loc_

        x=0
        if i == new_prop:          x = 1
        elif i == new_size:        x = 1
        elif i == 'superficie_m2': x = superficie_m2
        elif i == 'ambientes':     x = ambientes
        elif i == 'amenities':     x = amenities
        elif i == new_loc:        x = 1
        new_data.append(x)
        
    # acomodamos el shape
    new_data = [new_data]
    # estandarizamos los valores
    new_data = stdscaler_X.transform(new_data)
    new_data = pd.DataFrame(new_data)
    # renombramos las columnas para poder filtrar la lista de significativas
    new_data.columns = features_needed
    # filtramos las significativas obtenidas
    X_new = new_data.reindex(lista_betas,axis=1)

    global prediccion
    # predecimos
    prediccion = model2.predict(X_new)
    # desescalamos el y con stdscaler.inverse
    prediccion = np.round(np.exp(stdscaler_y.inverse_transform(prediccion)[0][0]),2)
    # elevando e a la predicción nos entrega el valor predicho
    

    return prediccion


###################################### FUNCION PREDICTORA ######################################
#-----------------------------------------------------------------------------------------------
########################################### SIDE BAR ###########################################

st.sidebar.subheader('CARACTERíSTICAS')

prop_ =  st.sidebar.selectbox('Tipo de Propiedad:',('PH','apartment','house','store'))
size_ =  st.sidebar.selectbox('Tamaño:',('normal','big','xl'))
superficie_m2 =  st.sidebar.text_input(label='Superficie total:',value=100)
ambientes =  st.sidebar.slider(label='Ambientes:',value=2,min_value=1,max_value=6)
if st.sidebar.checkbox("Amenities"):  amenities = 1
else:                                 amenities = 0


########################################### SIDE BAR ###########################################
#-----------------------------------------------------------------------------------------------
######################################## CONTROLADORES #########################################

st.title('Predictor de Precios - Properatti')
st.subheader('Desafío 2 - Grupo 1')

loc_ =  st.selectbox('Seleccione un barrio:',(lugares))


######################################## CONTROLADORES #########################################
#-----------------------------------------------------------------------------------------------
###################################### TABLA RESULTADOS ########################################

def resul():
    valores = []
    for i in lugares:
        valores.append(datos_nuevos(i))

    global res_map
    res_map = pd.DataFrame([lugares,valores]).T
    res_map.columns = ['BARRIO','USD_m2']
    res_map = res_map.merge(barrios)[['BARRIO','USD_m2','geometry']]
    res_map = gpd.GeoDataFrame(res_map,crs=crs,geometry='geometry')
    
    global resultados
    resultados = pd.DataFrame(valores,lugares)
    resultados.columns = ['usd_m2']
    resultados = resultados.sort_values('usd_m2',ascending=False)
    resultados['usd_m2'] = resultados.usd_m2.astype(int)
    resultados = resultados



###################################### TABLA RESULTADOS ########################################
#-----------------------------------------------------------------------------------------------
########################################## PREDICCION ##########################################

if amenities ==1:
    amen = 'Con Amenities'
else:
    amen = 'Sin Amenities'


if st.button('Predecir'):
    datos_nuevos(loc_)

    st.text('Propiedad: ' + prop_ + '\n' \
          + 'Tamaño relativo: ' + size_ + '\n' \
          + 'Tamaño absoluto: ' + str(superficie_m2) +' m2' + '\n' \
          + 'Ambientes: ' + str(ambientes) + '\n' \
          + 'Amenities: ' + amen + '\n' \
          + 'Barrio: ' + str(loc_))

    st.text('Valor predicho: ' + '\n' \
             + str(prediccion) + ' U$S por m2.')

    st.text('\nSegún su selección los siguientes son los barrios más caros.')


    resul()    
    #### MAPA
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    
    res_map.plot(ax=ax,column='USD_m2',cmap='Reds',edgecolor='lightgrey')
    plt.title('Valores predichos de m2 por barrio \n',size=20)
    plt.axis(False)
    plt.grid(False)

    st.pyplot(fig)
    

    #### BARRAS
    custom_pal = sns.color_palette("Reds_r", 48)
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(1,1,1)
    barras = res_map.sort_values('USD_m2',ascending=False)
    sns.barplot(x="USD_m2", y="BARRIO", data=barras.head(), palette=custom_pal,dodge=False)
    plt.ylabel(None)
    plt.xticks(size=11)
    plt.yticks(size=13,rotation=45)
    plt.xlabel('Precio por m2',size=14,labelpad=10);

    st.subheader('Top 5 barrios')
    st.pyplot(fig)
    