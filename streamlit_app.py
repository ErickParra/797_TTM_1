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
from datetime import datetime


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

# Diccionario para mapear nombres de columnas
column_mapping = {
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
def convert_units(data):
        # Conversión de presión: kPa a psi
        pressure_columns = [
            "Engine Oil Pressure (797F)",
            "Service Brake Accumulator Pressure (797F)",
            "Differential (Axle) Lube Pressure (797F)",
            "Steering Accumulator Oil Pressure (797F)",
            "Machine System Air Pressure (797F)",
            "Intake Manifold #2 Pressure (797F)",
            "Intake Manifold Pressure (797F)",
            "Left Rear Parking Brake Oil Pressure (797F)",
            "Fuel Pressure (797F)",
            "Right Rear Parking Brake Oil Pressure (797F)",
            "Oil Filter Differential Pressure (797F)",
            "Engine Coolant Pump Outlet Pressure (797F)",
            "Desired Fuel Rail Pressure (797F)",
            "Fuel Rail Pressure (797F)"
        ]
        for col in pressure_columns:
            if col in data.columns:
                data[col] = data[col] * 0.145038

        # Conversión de temperatura: °C a °F
        temperature_columns = [
            "Intake Manifold Air Temperature (797F)",
            "Intake Manifold #2 Air Temperature (797F)",
            "Right Exhaust Temperature (797F)",
            "Left Exhaust Temperature (797F)",
            "Left Front Brake Oil Temperature (797F)",
            "Right Front Brake Oil Temperature (797F)",
            "Right Rear Brake Oil Temperature (797F)",
            "Left Rear Brake Oil Temperature (797F)",
            "Engine Coolant Pump Outlet Temperature (797F)",
            "Engine Coolant Temperature (797F)",
            "Transmission Oil Temperature (797F)",
            "Engine Oil Temperature (797F)"
        ]
        for col in temperature_columns:
            if col in data.columns:
                data[col] = data[col] * 9 / 5 + 32

        return data

# Paso 1: Renombrar columnas
data.rename(columns=column_mapping, inplace=True)

# Paso 2: Convertir unidades
data = convert_units(data)

# Paso 3: Resamplear a 30 segundos
data["ReadTime"] = pd.to_datetime(data["ReadTime"])
data.set_index("ReadTime", inplace=True)

# Filtrar solo columnas numéricas antes de resamplear
numeric_columns = data.select_dtypes(include=["number"]).columns
resampled_data = data[numeric_columns].resample("30S").mean()

# Paso 4: Pivotear datos
pivoted_data = resampled_data.reset_index().pivot_table(
        index="ReadTime",
        columns="ParameterName",
        values="ParameterFloatValue"
    )

# Mostrar datos procesados
st.write("### Datos procesados y listos para TTM")
st.dataframe(pivoted_data)