import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from models import TimeSeriesPredictor_ModelWithDropout
from models import TimeSeriesPredictor, TimeSeriesPredictor_Model
from sklearn.preprocessing import MinMaxScaler


def custom_parser(date_str):
    months_mapping = {
        "ene": "Jan",
        "feb": "Feb",
        "mar": "Mar",
        "abr": "Apr",
        "may": "May",
        "jun": "Jun",
        "jul": "Jul",
        "ago": "Aug",
        "sep": "Sep",
        "oct": "Oct",
        "nov": "Nov",
        "dic": "Dec",
    }
    for es, en in months_mapping.items():
        date_str = date_str.replace(es, en)
    return pd.to_datetime(date_str, format='%d-%b-%y', dayfirst=True)


@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv('Precios.csv', parse_dates=['FECHA'],
                       date_parser=custom_parser)


df = get_data()


@st.cache_data
def get_test_data() -> pd.DataFrame:
    return df[(df['FECHA'] > '2023-03-31') & (df['FECHA'] <= '2023-06-30')]


test_data = get_test_data()


@st.cache_data
def list_ofModels():
    input_dim = 1
    hidden_dim = 50
    num_layers = 2
    seq_length = 10

    # Instanciamos el modelo
    modelA = TimeSeriesPredictor(input_dim, hidden_dim, seq_length, num_layers)
    model1 = TimeSeriesPredictor_Model(input_dim, 100, seq_length)
    model2 = TimeSeriesPredictor_ModelWithDropout(input_dim, 64,
                                                  seq_length, dropout_prob=0.2)
    state_names = ['modelA', 'model1', 'model2']
    state_ = [modelA, model1, model2]
    for idx, model in enumerate(state_):
        model.load_state_dict(torch.load(f'models/{state_names[idx]}.pth'))
    return state_
 

list_models = list_ofModels()


st.title("Dashboard de Precios de Combustible")
fuel_type = st.sidebar.selectbox("Selecciona el tipo de combustible",
                                 ['Todos', 'Superior', 'Regular', 'Diesel'])


st.subheader("Precios Mensuales de Combustible")
df['Mes'] = df['FECHA'].dt.month_name()
if fuel_type == 'Todos':
    fig1 = px.line(df, x='Mes', y=['Superior', 'Regular', 'Diesel'],
                   title='Precios de combustible por mes')
else:
    fig1 = px.line(df, x='Mes', y=fuel_type,
                   title=f'Precio de {fuel_type} por mes')
st.plotly_chart(fig1)


st.subheader("Tendencia de precios a lo largo del tiempo")
if fuel_type == 'Todos':
    fig2 = px.line(df, x='FECHA', y=['Superior', 'Regular', 'Diesel'],
                   title='Tendencia de precios a lo largo del tiempo')
else:
    fig2 = px.line(df, x='FECHA', y=fuel_type,
                   title=f'Tendencia de precios de {fuel_type} a lo largo del tiempo')
st.plotly_chart(fig2)

st.subheader("Predicciones de Precios")
model_choice = st.sidebar.selectbox("Selecciona un modelo",
                                    ['Fully Conected NR', 'LSTM',
                                     'LSTM con Droupt'])
x = df.index.values.reshape(-1, 1)

if fuel_type == 'Todos':
    y = df[['Superior', 'Regular', 'Diesel']]
else:
    y = df[[fuel_type]]


if model_choice == 'Fully Conected NR':
    model = list_models[0]
elif model_choice == 'LSTM':
    model = list_models[1]
else:
    model = list_models[2]

# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# st.write(f'Error Cuadrático Medio: {mean_squared_error(y_test, y_pred)}')


def predict_future(model, last_sequence, future_steps):
    future_predictions = []
    current_sequence = last_sequence.clone().detach()
    
    model.eval()
    with torch.no_grad():
        for i in range(future_steps):
            prediction = model(current_sequence)
            future_predictions.append(prediction.item())
            # Creamos una copia de la secuencia actual antes de actualizarla
            new_sequence = current_sequence.clone()
            new_sequence[0, :-1, 0] = current_sequence[0, 1:, 0]
            new_sequence[0, -1, 0] = prediction
            current_sequence = new_sequence
        
    return future_predictions


def get_predictions(last_sequence):
    future_steps = 90
    future_predictions = predict_future(model, last_sequence, future_steps)
    return future_predictions


# def plot_predictions(fuel, predictions):
    
#     # Asegúrate de obtener los datos de entrenamiento correctos
#     train_data_visualization = df[fuel].values

#     # Gráfica comparando los datos de entrenamiento
#     plt.figure(figsize=(15, 6))
#     plt.plot(train_data_visualization, label='Datos de Entrenamiento',
#              marker='o', color='blue')
    
#     plt.plot(range(len(train_data_visualization),
#                    len(train_data_visualization) + len(predictions)),
#              predictions,
#              label='Predicciones', marker='x',
#              color='orange')
    
#     plt.xlabel('Días desde el inicio de la serie')
#     plt.ylabel(f'Precio del Combustible {fuel}')
#     plt.title(f'Comparación de Datos de Entrenamiento, Predicciones y Datos Reales (2021 - 2023) para {fuel}')
#     plt.legend()
#     plt.grid(True)
#     st.pyplot(plt)  # Muestra la gráfica en Streamlit

def plot_predictions(fuel, predictions):
    
    # Create a copy of the original dataframe
    df_vis = df.copy()

    # Adding month name for visualization

    df_vis['Mes'] = df_vis['FECHA'].dt.month_name()
    # Create a new dataframe for the predictions
    df_predictions = pd.DataFrame({
        'FECHA': pd.date_range(df_vis['FECHA'].iloc[-1] + pd.Timedelta(days=1), periods=len(predictions)),
        'Predicciones': predictions.flatten()
    })

    # Append the predictions dataframe to the original dataframe
    df_vis = pd.concat([df_vis, df_predictions], axis=0)

    # Plot fuel price trend over time
    st.subheader(f"Predicciones para {fuel}")
    if fuel == 'Todos':
        fig2 = px.line(df_vis, x='FECHA', y=['Superior', 'Regular', 'Diesel', 'Predicciones'], 
                       title=f'Tendencia de precios de {fuel} a lo largo del tiempo')
    else:
        fig2 = px.line(df_vis, x='FECHA', y=[fuel, 'Predicciones'], 
                       title=f'Tendencia de precios de {fuel} a lo largo del tiempo')
    st.plotly_chart(fig2)

if fuel_type == 'Todos':
    for fuel in ['Superior', 'Regular', 'Diesel']:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = scaler.fit_transform(df[[fuel]])
        scaled_test_data = scaler.transform(test_data[[fuel]])
        last_sequence = torch.tensor(
            scaled_test_data[-11:-1].reshape(1, 10, 1),
            dtype=torch.float32)
        
        future_predictions_scaled = get_predictions(last_sequence)
        future_predictions = scaler.inverse_transform(
            np.array(future_predictions_scaled).reshape(-1, 1))

        plot_predictions(fuel, future_predictions)

else:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(df[[fuel_type]])
    scaled_test_data = scaler.transform(test_data[[fuel_type]])
    last_sequence = torch.tensor(
        scaled_test_data[-11:-1].reshape(1, 10, 1),
        dtype=torch.float32)
    future_predictions_scaled = get_predictions(last_sequence)
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions_scaled).reshape(-1, 1))
    
    plot_predictions(fuel_type, future_predictions)
