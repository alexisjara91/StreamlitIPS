import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo
model = load_model('modelo_entrenado.keras')

st.title("Predicciones con el Modelo de Deep Learning")

st.write("Sube un archivo CSV con los datos ya escalados para realizar predicciones:")

# Subir archivo CSV
uploaded_file = st.file_uploader("Elige un archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Leer el archivo CSV
        data = pd.read_csv(uploaded_file)

        # Verificar que los datos tengan el formato adecuado
        st.write("Datos cargados (limitados a los primeros 10 registros):")
        st.write(data.head(10))

        # Limitar a un máximo de 10 registros
        data_limited = data.head(10)

        # Convertir a un array de NumPy
        input_data = data_limited.to_numpy()

        # Realizar predicciones
        predictions = model.predict(input_data)

        # Crear un DataFrame con predicciones y los índices originales
        predicciones_df = pd.DataFrame({
            "Índice": data_limited.index,
            "Predicción": predictions.flatten()  # Asegurarse de que las predicciones sean 1D
        })

        # Mostrar resultados
        st.write("Predicciones realizadas:")
        st.write(predicciones_df)

        # Guardar las predicciones en un archivo CSV descargable
        st.download_button(
            label="Descargar predicciones",
            data=predicciones_df.to_csv(index=False),
            file_name="predicciones.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
