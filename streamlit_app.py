import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyodbc
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pyodbc
#from datetime import datetime

from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.visualization import plot_predictions
from datetime import datetime, timedelta
from databricks import sql

#from tsfm_public.models import TinyTimeMixerForPrediction
from transformers import AutoConfig
import torch

# Acceder a los secrets almacenados en Streamlit Cloud
server = st.secrets["server"]
database = st.secrets["database"]
username = st.secrets["username"]
password = st.secrets["password"]

@st.cache
def load_data(query, conn_str):
    try:
        conn = pyodbc.connect(conn_str)
        data = pd.read_sql(query, conn)
        conn.close()
        return data
    except pyodbc.Error as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        return pd.DataFrame()

# Configuración de la conexión a la base de datos
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Ejecución de la consulta SQL con todos los parámetros
query = """
SELECT
       [EquipmentName],
       [ReadTime],
       [EquipmentModel],
       [ParameterName],
       [ParameterFloatValue]
FROM [OemDataProvider].[OemParameterExternalView]
WHERE 
      [EquipmentModel] = '797F' AND 
      [EquipmentName] = 'C46' AND 
      [ParameterName] IN (
          'Parking Brake (797F)', 
          'Cold Mode (797F)', 
          'Shift Lever Position (797F)',
          'Oht Truck Payload State (797F)', 
          'Engine Oil Pressure (797F)', 
          'Service Brake Accumulator Pressure (797F)',
          'Differential (Axle) Lube Pressure (797F)', 
          'Steering Accumulator Oil Pressure (797F)',
          'Intake Manifold Air Temperature (797F)', 
          'Intake Manifold #2 Air Temperature (797F)', 
          'Machine System Air Pressure (797F)',
          'Intake Manifold #2 Pressure (797F)', 
          'Intake Manifold Pressure (797F)', 
          'Left Rear Parking Brake Oil Pressure (797F)',
          'Fuel Pressure (797F)', 
          'Right Rear Parking Brake Oil Pressure (797F)',
          'Transmission Input Speed (797F)', 
          'Engine Coolant Pump Outlet Pressure (797F)', 
          'Engine Speed (797F)',
          'Desired Fuel Rail Pressure (797F)', 
          'Fuel Rail Pressure (797F)', 
          'Engine Fan Speed (797F)',
          'Right Exhaust Temperature (797F)', 
          'Left Exhaust Temperature (797F)', 
          'Left Front Brake Oil Temperature (797F)',
          'Right Front Brake Oil Temperature (797F)', 
          'Oil Filter Differential Pressure (797F)', 
          'Right Rear Brake Oil Temperature (797F)',
          'Left Rear Brake Oil Temperature (797F)', 
          'Engine Coolant Pump Outlet Temperature (797F)', 
          'Engine Coolant Temperature (797F)',
          'Transmission Oil Temperature (797F)', 
          'Engine Oil Temperature (797F)'
      ) AND
      ParameterFloatValue IS NOT NULL AND 
      ReadTime > (DATEADD(HOUR, -120, GETDATE()))
"""

# Cargar datos
with st.spinner('Ejecutando consulta...'):
    data = load_data(query, conn_str)
st.success('Consulta completada!')

# Verificar si los datos son válidos
if data.empty:
    st.error("No se pudo obtener datos de la base de datos. Revisa la conexión y los logs para más detalles.")
else:
    st.write("### Datos obtenidos del equipo C46")
    st.dataframe(data)

    # Selector de parámetro para graficar
    selected_param = st.selectbox(
        "Seleccione un parámetro para graficar:",
        data['ParameterName'].unique()
    )



    # Filtrar datos para el parámetro seleccionado
    filtered_data = data[(data['ParameterName'] == selected_param) &
                     (data['ParameterFloatValue'] >= -100) &
                     (data['ParameterFloatValue'] <= 5000)]

    # Asegurarse de que ReadTime sea datetime y ordenar los datos
    filtered_data['ReadTime'] = pd.to_datetime(filtered_data['ReadTime'])
    filtered_data = filtered_data.sort_values(by='ReadTime')

    # Graficar datos
    st.write(f"### Gráfico de {selected_param} para el equipo C46")
    fig, ax = plt.subplots(figsize=(12, 6))

    if not filtered_data.empty:
        ax.plot(
            filtered_data['ReadTime'], 
            filtered_data['ParameterFloatValue'], 
            label=selected_param, 
            color='blue', 
            linewidth=1
        )
        ax.set_title(f"{selected_param} para C46")
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Valor")
        ax.legend()
        plt.grid()
        st.pyplot(fig)
    else:
        st.write(f"No hay datos disponibles para {selected_param} en el rango especificado (-100 a 5000).")




import pandas as pd
import numpy as np

# Verificar valores únicos en la columna ParameterName
st.write("### Valores únicos en ParameterName")
unique_parameters = data['ParameterName'].unique()  # Obtener valores únicos
st.write(f"Número de parámetros únicos: {len(unique_parameters)}")
st.write("#### Lista de parámetros únicos:")
st.write(unique_parameters)




# Asegúrate de que 'ReadTime' esté en formato datetime
data['ReadTime'] = pd.to_datetime(data['ReadTime'])

# Pivotear los datos para que cada ParameterName sea una columna
pivoted_data = data.pivot_table(
    index='ReadTime', 
    columns='ParameterName', 
    values='ParameterFloatValue'
)

# Resamplear los datos a intervalos de 30 segundos, llenando valores faltantes con interpolación
resampled_data = pivoted_data.resample('30S').mean().interpolate(method='linear')

# Mostrar los primeros registros para verificar
st.write("### Datos resampleados a 30 segundos")
st.dataframe(resampled_data)

# Guardar los datos procesados si es necesario
# resampled_data.to_csv("resampled_data.csv")


# Diccionario para mapear nombres de columnas (después del pivot)
vims_column_mapping = {
    "Parking Brake (797F)": "Parking Brake-Brake ECM ()",
    "Cold Mode (797F)": "Cold Mode-Engine ()",
    "Shift Lever Position (797F)": "Shift Lever Position-Chassis Ctrl ()",
    "Oht Truck Payload State (797F)": "OHT Truck Payload State-Communication Gateway #2 ()",
    "Engine Oil Pressure (797F)": "Engine Oil Pressure-Engine (psi)",
    "Service Brake Accumulator Pressure (797F)": "Service Brake Accumulator Pressure-Brake ECM (psi)",
    "Differential (Axle) Lube Pressure (797F)": "Differential (Axle) Lube Pressure-Brake ECM (psi)",
    "Steering Accumulator Oil Pressure (797F)": "Steering Accumulator Oil Pressure-Chassis Ctrl (psi)",
    "Intake Manifold Air Temperature (797F)": "Intake Manifold Air Temperature-Engine (Deg F)",
    "Intake Manifold #2 Air Temperature (797F)": "Intake Manifold #2 Air Temperature-Engine (Deg F)",
    "Machine System Air Pressure (797F)": "Machine System Air Pressure-Chassis Ctrl (psi)",
    "Intake Manifold #2 Pressure (797F)": "Intake Manifold #2 Pressure-Engine (psi)",
    "Intake Manifold Pressure (797F)": "Intake Manifold Pressure-Engine (psi)",
    "Left Rear Parking Brake Oil Pressure (797F)": "Left Rear Parking Brake Oil Pressure-Brake ECM (psi)",
    "Fuel Pressure (797F)": "Fuel Pressure-Engine (psi)",
    "Right Rear Parking Brake Oil Pressure (797F)": "Right Rear Parking Brake Oil Pressure-Brake ECM (psi)",
    "Transmission Input Speed (797F)": "Transmission Input Speed-Trans Ctrl (rpm)",
    "Engine Coolant Pump Outlet Pressure (797F)": "Engine Coolant Pump Outlet Pressure (absolute)-Engine (psi)",
    "Engine Speed (797F)": "Engine Speed-Engine (rpm)",
    "Desired Fuel Rail Pressure (797F)": "Desired Fuel Rail Pressure-Engine (psi)",
    "Fuel Rail Pressure (797F)": "Fuel Rail Pressure-Engine (psi)",
    "Engine Fan Speed (797F)": "Engine Fan Speed-Brake ECM (rpm)",
    "Right Exhaust Temperature (797F)": "Right Exhaust Temperature-Engine (Deg F)",
    "Left Exhaust Temperature (797F)": "Left Exhaust Temperature-Engine (Deg F)",
    "Left Front Brake Oil Temperature (797F)": "Left Front Brake Oil Temperature-Brake ECM (Deg F)",
    "Right Front Brake Oil Temperature (797F)": "Right Front Brake Oil Temperature-Brake ECM (Deg F)",
    "Oil Filter Differential Pressure (797F)": "Oil Filter Differential Pressure-Engine (psi)",
    "Right Rear Brake Oil Temperature (797F)": "Right Rear Brake Oil Temperature-Brake ECM (Deg F)",
    "Left Rear Brake Oil Temperature (797F)": "Left Rear Brake Oil Temperature-Brake ECM (Deg F)",
    "Engine Coolant Pump Outlet Temperature (797F)": "Engine Coolant Pump Outlet Temperature-Engine (Deg F)",
    "Engine Coolant Temperature (797F)": "Engine Coolant Temperature-Engine (Deg F)",
    "Transmission Oil Temperature (797F)": "Transmission Oil Temperature-Trans Ctrl (Deg F)",
    "Engine Oil Temperature (797F)": "Engine Oil Temperature-Engine (Deg F)"
}

# Conversión de unidades
def convert_units(resampled_data):
    # Conversión de presión: kPa a psi
    pressure_columns = [
        "Engine Oil Pressure-Engine (psi)",
        "Service Brake Accumulator Pressure-Brake ECM (psi)",
        "Differential (Axle) Lube Pressure-Brake ECM (psi)",
        "Steering Accumulator Oil Pressure-Chassis Ctrl (psi)",
        "Machine System Air Pressure-Chassis Ctrl (psi)",
        "Intake Manifold #2 Pressure-Engine (psi)",
        "Intake Manifold Pressure-Engine (psi)",
        "Left Rear Parking Brake Oil Pressure-Brake ECM (psi)",
        "Fuel Pressure-Engine (psi)",
        "Right Rear Parking Brake Oil Pressure-Brake ECM (psi)",
        "Oil Filter Differential Pressure-Engine (psi)",
        "Engine Coolant Pump Outlet Pressure (absolute)-Engine (psi)",
        "Desired Fuel Rail Pressure-Engine (psi)",
        "Fuel Rail Pressure-Engine (psi)"
    ]
    for col in pressure_columns:
        if col in resampled_data.columns:
            resampled_data[col] = resampled_data[col] * 0.145038

    # Conversión de temperatura: °C a °F
    temperature_columns = [
        "Intake Manifold Air Temperature-Engine (Deg F)",
        "Intake Manifold #2 Air Temperature-Engine (Deg F)",
        "Right Exhaust Temperature-Engine (Deg F)",
        "Left Exhaust Temperature-Engine (Deg F)",
        "Left Front Brake Oil Temperature-Brake ECM (Deg F)",
        "Right Front Brake Oil Temperature-Brake ECM (Deg F)",
        "Right Rear Brake Oil Temperature-Brake ECM (Deg F)",
        "Left Rear Brake Oil Temperature-Brake ECM (Deg F)",
        "Engine Coolant Pump Outlet Temperature-Engine (Deg F)",
        "Engine Coolant Temperature-Engine (Deg F)",
        "Transmission Oil Temperature-Trans Ctrl (Deg F)",
        "Engine Oil Temperature-Engine (Deg F)"
    ]
    for col in temperature_columns:
        if col in resampled_data.columns:
            resampled_data[col] = resampled_data[col] * 9 / 5 + 32

    return resampled_data

# Renombrar columnas
resampled_data.rename(columns=vims_column_mapping, inplace=True)

# Aplicar conversiones de unidades
resampled_data = convert_units(resampled_data)

import pytz

# Ajustar ReadTime a la zona horaria de Santiago, Chile
resampled_data.index = resampled_data.index.tz_localize('UTC').tz_convert('America/Santiago')

# Ordenar los datos por ReadTime en orden descendente
resampled_data = resampled_data.sort_index(ascending=False)

# Mostrar datos procesados
st.write("### Datos procesados después de conversiones y renombrados")
st.dataframe(resampled_data)




import os
import joblib
from transformers import AutoConfig, AutoModelForCausalLM

import json
import streamlit as st

# Ruta al archivo config.json
config_path = "./config.json"

# Función para cargar y mostrar el archivo de configuración
def display_config_file(config_path):
    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)
        
        # Mostrar el contenido del archivo en Streamlit
        st.write("### Contenido del archivo config.json")
        st.json(config_data)  # Usa st.json para un formato legible
    except FileNotFoundError:
        st.error(f"El archivo {config_path} no se encontró.")
    except json.JSONDecodeError as e:
        st.error(f"Error al leer el archivo JSON: {e}")
    except Exception as e:
        st.error(f"Error inesperado: {e}")

# Llamar a la función para mostrar el archivo
display_config_file(config_path)


#git clone https://github.com/ibm-granite/granite-tsfm.git
#cd granite-tsfm
#pip install .


# Función para cargar el modelo TTM
# Function to load the TTM model

# Paths to the model and configuration
MODEL_PATH = "./model.safetensors"
#CONFIG_PATH = "./config.json"
#Configuración de paths
MODEL_DIR = "."  # Directorio actual donde están los archivos
#CONFIG_PATH = f"{MODEL_DIR}/config.json"
OBSERVABLE_SCALER_PATH = "./observable_scaler_0.pkl"
TARGET_SCALER_PATH = "./target_scaler_0.pkl"



# Function to load the TTM model
@st.cache_resource
def load_ttm_model():
    try:
        # Load configuration
        config = TinyTimeMixerForPrediction.from_pretrained(config_path)
        
        # Load the model
        model = TinyTimeMixerForPrediction.from_pretrained(
            pretrained_model_name_or_path=MODEL_PATH,
            config=config,
            from_tf=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None




from transformers import AutoConfig, PreTrainedModel
import torch
import joblib
import os


# Función para cargar el modelo
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        # Cargar el modelo desde el directorio donde están los archivos
        model = TinyTimeMixerForPrediction.from_pretrained(
            pretrained_model_name_or_path=MODEL_DIR,  # Directorio que contiene los archivos
            config=config_path  # Ruta al archivo de configuración
        )
        st.success("Modelo TTM cargado correctamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo TTM: {e}")
        return None

# Llamada para cargar el modelo
model = load_model()

# Verificación de la carga del modelo
if model is not None:
    st.write("### Detalles del modelo cargado")
    st.write(model)
else:
    st.error("No se pudo cargar el modelo. Revisa los archivos y las configuraciones.")


# Función para cargar los escaladores
@st.cache_resource
def load_scalers():
    try:
        observable_scaler = joblib.load(OBSERVABLE_SCALER_PATH)
        target_scaler = joblib.load(TARGET_SCALER_PATH)
        return observable_scaler, target_scaler
    except Exception as e:
        st.error(f"Error al cargar los escaladores: {e}")
        return None, None

# Cargar el modelo y escaladores
model = load_model()
observable_scaler, target_scaler = load_scalers()

# Verificar la carga exitosa
if model is not None:
    st.success("Modelo TTM cargado correctamente.")
else:
    st.error("No se pudo cargar el modelo TTM. Verifica los archivos.")

if observable_scaler is not None and target_scaler is not None:
    st.success("Escaladores cargados correctamente.")
else:
    st.error("No se pudieron cargar los escaladores.")

# Mostrar detalles de la configuración
st.write("### Configuración del Modelo")
try:
    with open(config_path, "r") as f:
        config = f.read()
        st.json(config)
except FileNotFoundError:
    st.error("Archivo de configuración no encontrado.")

# Simular predicción con datos de entrada
st.write("### Simulación de predicciones")
uploaded_file = st.file_uploader("Sube un archivo CSV con datos de entrada", type="csv")
if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    st.write("Datos de entrada cargados:")
    st.dataframe(input_data)

    try:
        # Normalizar los datos de entrada con el observable scaler
        normalized_data = observable_scaler.transform(input_data.values)

        # Convertir datos a tensor para alimentar al modelo
        input_tensor = torch.tensor(normalized_data).unsqueeze(0)

        # Generar predicciones
        with torch.no_grad():
            predictions = model(input_tensor)

        # Desescalar las predicciones
        descaled_predictions = target_scaler.inverse_transform(predictions.numpy().squeeze(0))
        st.write("Predicciones desescaladas:")
        st.dataframe(descaled_predictions)
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")


# Ajustar el índice y renombrar columna si es necesario
resampled_data = resampled_data.reset_index()  # Restaurar 'ReadTime' como una columna
if 'ReadTime' in resampled_data.columns:
    resampled_data.rename(columns={'ReadTime': 'New_Date/Time'}, inplace=True)



# Inspeccionar las columnas y formatos en Streamlit
st.write("### Inspección de columnas y formatos")

# Mostrar los nombres de las columnas
st.write("#### Nombres de las columnas")
st.write(resampled_data.columns.tolist())

# Mostrar los tipos de datos de las columnas
st.write("#### Tipos de datos de las columnas")
st.write(resampled_data.dtypes)

# Mostrar los primeros registros del DataFrame
st.write("#### Primeros registros del DataFrame")
st.dataframe(resampled_data.head())

# Resumen estadístico de las columnas numéricas
st.write("#### Resumen estadístico de las columnas numéricas")
st.dataframe(resampled_data.describe())

# Verificar si hay valores nulos
st.write("#### Verificación de valores nulos")
st.write(resampled_data.isnull().sum())



# Definir la columna target
target_col = "Engine Oil Temperature-Engine (Deg F)"  # Cambiar si el target es diferente

# Verificar si la columna target está en los datos
if target_col not in resampled_data.columns:
    st.error(f"La columna target '{target_col}' no está presente en los datos.")
else:
    try:
        # Separar features, target y tiempo
        time_col = "New_Date/Time"  # Columna con las fechas
        timestamps = resampled_data[time_col]  # Timestamps
        X = resampled_data.drop(columns=[time_col])  # Features con target incluido

        # Ajustar para los últimos 512 valores más recientes
        context_length = 512  # Longitud esperada por el modelo
        if len(X) > context_length:
            X = X[-context_length:]
            timestamps = timestamps[-context_length:]

        # Escalar las features asegurando el tipo float32
        if target_col in X.columns:
            X_for_scaler = X.drop(columns=[target_col])  # Excluir la columna target para escalar
        else:
            X_for_scaler = X

        normalized_X = observable_scaler.transform(X_for_scaler.values.astype('float32'))

        # Agregar nuevamente la columna target como parte del input normalizado
        if target_col in X.columns:
            normalized_X = np.hstack(
                [normalized_X, X[target_col].values.reshape(-1, 1).astype('float32')]
            )

        # Convertir a tensor asegurando el tipo float32
        input_tensor = torch.tensor(normalized_X, dtype=torch.float32).unsqueeze(0)

        # Generar predicciones
        with torch.no_grad():
            model_output = model(input_tensor)  # Obtener la salida del modelo
            predictions_tensor = model_output.prediction  # Extraer el tensor de predicciones

        # Convertir el tensor de predicciones a un array numpy
        descaled_predictions = target_scaler.inverse_transform(predictions_tensor.cpu().numpy().squeeze(0))

        # Crear un DataFrame con las predicciones y el tiempo
        predictions_df = pd.DataFrame({
            "New_Date/Time": timestamps.values,  # Timestamps para las predicciones
            "Predicción": descaled_predictions.flatten()  # Predicciones desescaladas
        })

        # Mostrar resultados
        st.write("### Resultados de las predicciones")
        st.dataframe(predictions_df)

        # Graficar resultados
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timestamps, predictions_df["Predicción"], label="Predicción", color="red", linestyle="--")
        ax.set_title("Predicción")
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Temperatura (Deg F)")
        ax.legend()
        plt.grid()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")



st.write(f"Dimensiones de las features antes de escalar: {X_for_scaler.shape}")
st.write(f"Dimensiones de las features después de escalar: {normalized_X.shape}")
