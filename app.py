import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('modelo_entrenado.h5')

st.title("Predicciones con el Modelo de Deep Learning")

st.write("Ingrese los datos requeridos para realizar la predicción NOCOBRO:")

# Ejemplo: si necesitas que el usuario ingrese PAGOS, MONTO, COMUNA, REGION, etc.
pagos = st.number_input("PAGOS:", value=0.0)
monto = st.number_input("MONTO:", value=0.0)
comuna = st.number_input("COMUNA:", value=0.0)
region = st.number_input("REGION:", value=0.0)
urbanidad = st.number_input("URBANIDAD:", value=0.0)
f_pago = st.number_input("FPAGO:", value=0.0)
tipo_beneficio = st.number_input("TIPOBENEFICIO:", value=0.0)
cobro_marzo = st.number_input("COBROMARZO:", value=0.0)
fec_nac = st.number_input("FEC_NAC:", value=0.0)
sexo = st.number_input("SEXO:", value=0.0)
ecivil = st.number_input("ECIVIL:", value=0.0)
nacionalidad = st.number_input("NACIONALIDAD:", value=0.0)

# Cuando el usuario presione el botón, se realiza la predicción
if st.button("Realizar Predicción"):
    # Crear el array con las entradas. Ajusta el orden y cantidad de features a tu modelo.
    # Suponiendo que el orden sea: [PAGOS, MONTO, COMUNA, REGION, URBANIDAD, FPAGO, TIPOBENEFICIO, COBROMARZO, FEC_NAC, SEXO, ECIVIL, NACIONALIDAD]
    input_array = np.array([[pagos, monto, comuna, region, urbanidad, f_pago, tipo_beneficio, 
                             cobro_marzo, fec_nac, sexo, ecivil, nacionalidad]])
    
    # Realizar la predicción
    prediction = model.predict(input_array)
    st.write("Predicción:", prediction[0])
