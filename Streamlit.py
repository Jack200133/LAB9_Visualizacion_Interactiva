import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
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


months_es = {
    1: 'Enero',
    2: 'Febrero',
    3: 'Marzo',
    4: 'Abril',
    5: 'Mayo',
    6: 'Junio',
    7: 'Julio',
    8: 'Agosto',
    9: 'Septiembre',
    10: 'Octubre',
    11: 'Noviembre',
    12: 'Diciembre'
}

df['Mes'] = df['FECHA'].dt.month.map(months_es)

# Orden de los meses
months_order = ['Enero', 'Febrero', 'Marzo',
                'Abril', 'Mayo', 'Junio', 'Julio',
                'Agosto', 'Septiembre', 'Octubre',
                'Noviembre', 'Diciembre']

# Asigna un tipo categórico con el orden correcto a la columna 'Mes'
df['Mes'] = pd.Categorical(df['Mes'], categories=months_order, ordered=True)

# Agrupa por mes y calcula la media de los precios de cada tipo de combustible
df_avg = df.groupby('Mes').agg(
    {'Superior': 'mean', 'Regular': 'mean', 'Diesel': 'mean'}).reset_index()

# Ahora la columna 'Mes' tiene un orden, entonces cuando grafiques,
# los meses estarán en el orden correcto

colores = {
    "Superior": "#C2272D",
    "Regular": "#7D232D",
    "Diesel": "#FBB039",
}

predicciones = {
    "Superior": "#FBB039",
    "Regular": "#C2272D",
    "Diesel": "#7D232D",
}


st.subheader("Precios Mensuales Promedio de Combustible")
if fuel_type == 'Todos':
    # Extraer los valores del diccionario y pasarlos como lista
    fig1 = px.line(df_avg, x='Mes', y=['Superior', 'Regular', 'Diesel'],
                   title='Precios promedio de combustible por mes',
                   color_discrete_sequence=list(colores.values()))
else:
    # Obtener el color del diccionario usando fuel_type como clave
    fig1 = px.line(df_avg, x='Mes', y=fuel_type,
                   title=f'Precio promedio de {fuel_type} por mes',
                   color_discrete_sequence=[colores[fuel_type]])
st.plotly_chart(fig1)


st.subheader("Tendencia de precios a lo largo del tiempo")
if fuel_type == 'Todos':
    fig2 = px.line(df, x='FECHA', y=['Superior', 'Regular', 'Diesel'],
                   title='Tendencia de precios a lo largo del tiempo',
                   color_discrete_sequence=list(colores.values()))
else:
    title = f"Tendencia de precios de {fuel_type} a lo largo del tiempo"
    fig2 = px.line(df, x='FECHA', y=fuel_type,
                   title=title,
                   color_discrete_sequence=[colores[fuel_type]])
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


def plot_predictions(fuel, predictions):
    
    # Create a copy of the original dataframe
    df_vis = df.copy()

    # Adding month name for visualization

    df_vis['Mes'] = df_vis['FECHA'].dt.month_name()
    # Create a new dataframe for the predictions
    df_predictions = pd.DataFrame({
        'FECHA': pd.date_range(df_vis['FECHA'].iloc[-1] + pd.Timedelta(days=1),
                               periods=len(predictions)),
        'Predicciones': predictions.flatten()
    })

    # Append the predictions dataframe to the original dataframe
    df_vis = pd.concat([df_vis, df_predictions], axis=0)

    # Plot fuel price trend over time
    st.subheader(f"Predicciones para {fuel}")
    title = f'Tendencia de precios de {fuel} a lo largo del tiempo'
    if fuel == 'Todos':

        fig2 = px.line(df_vis, x='FECHA',
                       y=['Superior', 'Regular', 'Diesel', 'Predicciones'],
                       title=title,
                       color_discrete_sequence=list(colores.values()))
    else:
        fig2 = px.line(df_vis, x='FECHA', y=[fuel, 'Predicciones'],
                       title=title,
                       color_discrete_sequence=[colores[fuel],
                                                predicciones[fuel]])
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
