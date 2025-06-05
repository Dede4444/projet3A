import pandas as pd
import numpy as np
from prophet import Prophet
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(layout="wide", page_title="Prévision du prix du blé")

st.title("📈 Prévision du prix du blé avec Prophet")
st.markdown("Ce modèle utilise la production mondiale et le taux de change EUR/USD pour prédire le prix du blé.")

# Chargement des données
uploaded_file = st.file_uploader("📤 Charger le fichier Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Vérification des colonnes attendues
    required_cols = {"Date", "prix_ble", "taux_ED", "prod_monde"}
    if not required_cols.issubset(df.columns):
        st.error(f"Le fichier doit contenir les colonnes : {', '.join(required_cols)}")
        st.stop()

    # Préparation des données
    df["Date"] = pd.to_datetime(df["Date"])
    df["prix_ble_lisse"] = df["prix_ble"].rolling(window=3, center=True).mean()
    df = df.dropna(subset=["prix_ble_lisse", "taux_ED", "prod_monde"])

    # Création du DataFrame pour Prophet
    df_model = pd.DataFrame({
        "ds": df["Date"],
        "y": df["prix_ble_lisse"],
        "taux_ED": df["taux_ED"],
        "prod_monde": df["prod_monde"]
    })

    # Choix de l'utilisateur : nombre de mois à prédire
    periods = st.slider("🔮 Nombre de mois à prédire :", min_value=6, max_value=36, value=12, step=6)

    # Initialisation du modèle Prophet
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

    # Création des dates futures
    future = model.make_future_dataframe(periods=periods, freq="M")
    last_taux = df_model["taux_ED"].iloc[-1]
    last_prod = df_model["prod_monde"].iloc[-1]
    future["taux_ED"] = last_taux
    future["prod_monde"] = last_prod

    # Prédiction
    forecast = model.predict(future)

    # Affichage du graphique principal
    st.subheader("📉 Graphique de prévision")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Composantes de la prédiction
    st.subheader("🔍 Analyse des composantes du modèle")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Tableau des prévisions
    st.subheader("📋 Tableau des valeurs prédites")
    forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
    forecast_display.columns = ["Date", "Prévision", "Borne basse", "Borne haute"]
    forecast_display["Date"] = forecast_display["Date"].dt.strftime("%Y-%m")
    st.dataframe(forecast_display.reset_index(drop=True), use_container_width=True)

    # Matrice de corrélation
    st.subheader("📊 Corrélation entre les variables")
    plt.figure(figsize=(6, 4))
    corr = df[["prix_ble", "taux_ED", "prod_monde"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    st.pyplot(plt.gcf())
