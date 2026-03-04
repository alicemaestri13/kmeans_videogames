
import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
import os

# Configurazione iniziale della pagina
st.set_page_config(page_title="Game Recommender", page_icon="🎮", layout="wide")

st.title("🎮 Sistema di Raccomandazione Videogiocatori")
st.write("Trova il tuo prossimo gioco preferito grazie al Machine Learning!")

# --- 1. CARICAMENTO DATI E MODELLO ---

# Calcola il percorso esatto della cartella in cui si trova app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model_and_features():
    # Unisce il percorso della cartella al nome del file
    percorso_knn = os.path.join(BASE_DIR, 'knn_model.pkl')
    percorso_features = os.path.join(BASE_DIR, 'features_scaled.pkl')
    
    knn = joblib.load(percorso_knn)
    features = joblib.load(percorso_features)
    return knn, features

@st.cache_data
def load_dataframe():
    percorso_df = os.path.join(BASE_DIR, 'dataframe_pulito.pkl')
    return pd.read_pickle(percorso_df)

knn_model, features_scaled = load_model_and_features()
df = load_dataframe()
# --- 2. NUOVO MENU DI NAVIGAZIONE A BOTTONI ---
scelta = option_menu(
    menu_title=None,
    options=["Esplorazione Dati", "Trova Giochi Simili", "Come Funziona"],
    icons=["bar-chart-line", "controller", "lightbulb"],
    default_index=1,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f0f2f6"},
        "icon": {"color": "#ff4b4b", "font-size": "25px"},
        "nav-link": {"font-size": "18px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#ff4b4b", "color": "white"},
    }
)

st.write("---")

# --- 3. CONTENUTI DELLE SCHEDE (Ora usiamo 'if' invece di 'with tab:') ---

if scelta == "Esplorazione Dati":
    st.header("Esplorazione del Dataset")
    numero_giochi = len(df)
    st.info(f"""
    **🎮 Il Dataset in pillole:**
    
    Per questo progetto abbiamo utilizzato il famoso dataset **'Video Game Sales with Ratings'** tratto da Kaggle. 
    Dopo un'accurata fase di pulizia per rimuovere i dati incompleti, il nostro motore di Machine Learning 
    può contare su un solido database di ben **{numero_giochi} videogiochi unici**!
    """)
    st.write("Ecco un'anteprima dei dati:")
    st.dataframe(df.head(10))
    st.subheader("I 10 Generi più popolari")
    st.bar_chart(df['Genre'].value_counts().head(10))

elif scelta == "Trova Giochi Simili":
    st.header("Il tuo Personal Shopper Videoludico")
    lista_giochi = sorted(df['Name'].unique())
    gioco_scelto = st.selectbox(
        "Scegli un gioco che hai amato (inizia a digitare per cercare):",
        lista_giochi,
        index=None,
        placeholder="Es. Grand Theft Auto V, Skyrim, FIFA..."
    )
    
    if st.button("Raccomandami giochi simili!", type="primary"):
        if gioco_scelto is None:
            st.warning("⚠️ Per favore, digita e seleziona un gioco prima di cliccare!")
        else:
            idx = df[df['Name'] == gioco_scelto].index[0]
            distanze, indici = knn_model.kneighbors([features_scaled[idx]])
            
            st.success(f"Ottima scelta! Se ti è piaciuto **{gioco_scelto}**, ecco 5 giochi che adorerai:")
            for i in range(1, len(indici[0])):
                indice_vicino = indici[0][i]
                nome = df.iloc[indice_vicino]['Name']
                piattaforma = df.iloc[indice_vicino]['Platform']
                genere = df.iloc[indice_vicino]['Genre']
                score = df.iloc[indice_vicino]['Critic_Score']
                st.info(f"**{i}. {nome}** | Piattaforma: {piattaforma} | Genere: {genere} | Voto Critica: {score}/100")

elif scelta == "Come Funziona":
    st.header("Dietro le quinte: K-Nearest Neighbors")
    st.write("""
    Questa applicazione utilizza l'algoritmo di Machine Learning **K-Nearest Neighbors (KNN)**.
    
    Immagina che ogni videogioco sia un punto nello spazio. L'algoritmo analizza le caratteristiche di ogni gioco 
    (genere, piattaforma, vendite, recensioni) e calcola la distanza matematica tra questi punti.
    
    Quando selezioni un gioco, il modello cerca i **5 punti più vicini** nello spazio multidimensionale, 
    proponendoti i titoli matematicamente più simili!
    """)
