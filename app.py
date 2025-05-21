import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Titre de l'application
st.title("Visualisation des prix du blé (données synthétiques)")

# Génération des données synthétiques
np.random.seed(42)  # pour reproductibilité
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
prix_ble = 200 + np.cumsum(np.random.normal(loc=0, scale=2, size=100))  # base à 200€/t

df = pd.DataFrame({
    "Date": dates,
    "Prix (€ / tonne)": prix_ble
})

# Affichage des données
st.subheader("Données du prix du blé")
st.dataframe(df)

# Affichage du graphique
st.subheader("Évolution du prix du blé")
st.line_chart(df.set_index("Date"))
