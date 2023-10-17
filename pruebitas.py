import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# data = pd.read_csv('Precios.csv')


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


data = get_data()

# Dashboard title
st.title("Precios de Combustibles")

# Asegurarse de que los datos estén ordenados por fecha
data.sort_values('FECHA', inplace=True)

# Extraer el mes y el año de la fecha
data['Mes'] = data['FECHA'].dt.strftime('%B %Y')

# Agrupar los datos por mes y obtener el promedio de los precios
monthly_data = data.groupby('Mes').mean()


def main():
    st.title("Dashboard de Combustibles")

    # Selector de datos a mostrar
    selected_data = st.multiselect(
        "Selecciona los combustibles a visualizar",
        options=["Superior", "Regular", "Diesel"],
        default=["Superior", "Regular", "Diesel"]
    )

    # Gráfico 1: Promedio mensual
    st.subheader("Promedio Mensual")
    fig1, ax1 = plt.subplots()
    monthly_data[selected_data].plot(kind='bar', ax=ax1)
    ax1.set_ylabel('Precio')
    ax1.set_xlabel('Mes')
    st.pyplot(fig1)

    # Gráfico 2: Serie Temporal
    st.subheader("Serie Temporal")
    fig2, ax2 = plt.subplots()
    data.set_index('FECHA')[selected_data].plot(ax=ax2)
    ax2.set_ylabel('Precio')
    st.pyplot(fig2)


if __name__ == "__main__":
    main()
