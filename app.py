import pandas as pd
import numpy as np
from prophet import Prophet
import streamlit as st
import matplotlib.pyplot as plt

# Configuration Streamlit
st.set_page_config(page_title="Prévision du prix du blé", layout="wide")

st.title("🌾 Prévision du Prix du Blé avec Prophet")
st.markdown("""
Bienvenue sur cette application de prévision du **prix du blé** 📉 basée sur le modèle Prophet de Meta AI.
Nous utilisons les données historiques du prix du blé, le **taux de change EUR/USD** et la **production mondiale annuelle de blé** comme variables explicatives.

Vous pouvez ici :
- Visualiser l'évolution historique du prix du blé
- Prédire les prix pour les mois à venir
- Explorer les tendances sous-jacentes
""")

# 1. Chargement des données directement depuis GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/<TON-UTILISATEUR>/<TON-DEPOT>/main/Donné_F_Prophet.xlsx"
    return pd.read_excel(url)

try:
    df = load_data()
except:
    st.error("❌ Erreur de chargement du fichier. Vérifie que le fichier Excel est bien dans ton repo GitHub.")
    st.stop()

# 2. Préparation des données
df["Date"] = pd.to_datetime(df["Date"])
df["prix_ble_lisse"] = df["prix_ble"].rolling(window=3, center=True).mean()
df = df.dropna(subset=["prix_ble_lisse", "taux_ED", "prod_monde"])

# 3. Format pour Prophet
df_model = pd.DataFrame({
    "ds": df["Date"],
    "y": df["prix_ble_lisse"],
    "taux_ED": df["taux_ED"],
    "prod_monde": df["prod_monde"]
})

# 4. Choix du nombre de mois à prédire
periods = st.slider("🔮 Nombre de mois à prédire :", min_value=6, max_value=36, value=12, step=6)

# 5. Initialisation du modèle
model = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=0.1,
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)
model.add_regressor("taux_ED")
model.add_regressor("prod_monde")
model.fit(df_model)

# 6. Création des dates futures
future = model.make_future_dataframe(periods=periods, freq="M")
future["taux_ED"] = df_model["taux_ED"].iloc[-1]
future["prod_monde"] = df_model["prod_monde"].iloc[-1]

# 7. Prédiction
forecast = model.predict(future)

# 8. Affichage du graphique de prévision
st.subheader("📈 Évolution prévue du prix du blé")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# 9. Composantes de la prévision
st.subheader("🔍 Composantes de la prévision")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# 10. Tableau des prédictions
st.subheader("📋 Tableau des prévisions")
forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
forecast_display.columns = ["Date", "Prévision", "Borne basse", "Borne haute"]
forecast_display["Date"] = forecast_display["Date"].dt.strftime("%Y-%m")
st.dataframe(forecast_display.reset_index(drop=True), use_container_width=True)

# 11. Footer
st.markdown("---")
st.markdown("**🧠 Modèle utilisé :** Prophet de Meta | **Données :** prix du blé, taux de change, production mondiale | **Auteur :** Ton Nom / Projet Étudiant / 2025")
