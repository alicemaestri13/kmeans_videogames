
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
    options=["Esplorazione Dati", "Trova Giochi Simili", "Clustering K-Means", "Come Funziona"],
    icons=["bar-chart-line", "controller", "pie-chart", "lightbulb"],
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
elif scelta == "Clustering K-Means":
    st.header("Analisi dei Gruppi (K-Means Clustering)")
    
    # Layout a due colonne che simula la dashboard della tua amica
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configurazione")
        # Slider per scegliere quanti giochi pescare dal dataset
        n_giochi = st.slider("Numero di giochi nel dataset", min_value=100, max_value=len(df), value=500, step=100)
        seed_campionamento = st.slider("Seed per campionamento", min_value=0, max_value=100, value=42)
        
        st.write("---")
        st.subheader("Caratteristiche per il clustering")
        # Checkbox per far scegliere all'utente quali dati usare
        usa_voto = st.checkbox("Voto Critica", value=True)
        usa_vendite = st.checkbox("Vendite Globali", value=True)
        usa_anno = st.checkbox("Anno di uscita", value=False)
        
        st.write("---")
        st.subheader("Parametri K-Means")
        k = st.slider("Numero di cluster (k)", min_value=2, max_value=8, value=5)
        random_seed = st.slider("Seed casuale per K-means", min_value=0, max_value=100, value=42)

    with col2:
        # 1. Peschiamo un campione casuale di giochi
        df_cluster = df.sample(n=n_giochi, random_state=seed_campionamento).copy()
        
        # 2. Capiamo quali caratteristiche ha spuntato l'utente
        feature_cols = []
        if usa_voto: feature_cols.append('Critic_Score')
        if usa_vendite: feature_cols.append('Global_Sales')
        if usa_anno: feature_cols.append('Year_of_Release')
        
        # Controlliamo che ne abbia scelte almeno due per poter fare il grafico
        if len(feature_cols) < 2:
            st.warning("⚠️ Seleziona almeno due caratteristiche dalla barra laterale per visualizzare il grafico.")
        else:
            X = df_cluster[feature_cols]
            
            # 3. IL TRUCCO MAGICO: Standardizziamo i dati!
            from sklearn.preprocessing import StandardScaler
            scaler_kmeans = StandardScaler()
            X_scaled = scaler_kmeans.fit_transform(X)
            
            # 4. Applichiamo il K-Means
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init='auto')
            df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Formattiamo i cluster come testo per colorarli bene
            df_cluster['Cluster'] = df_cluster['Cluster'].astype(str)
            
            # Scegliamo cosa mettere sugli assi X e Y (le prime due spunte selezionate)
            asse_x = feature_cols[0]
            asse_y = feature_cols[1]
            
            # 5. Creiamo il grafico!
            import plotly.express as px
            fig = px.scatter(
                df_cluster,
                x=asse_x,
                y=asse_y,
                color="Cluster",
                hover_name="Name",
                hover_data=["Genre", "Platform"],
                title=f"Distribuzione Videogiochi: {asse_x} vs {asse_y}",
                color_discrete_sequence=px.colors.qualitative.Pastel # Colori più tenui e professionali
            )
            
            # Rimuoviamo lo sfondo per renderlo più pulito come quello della tua amica
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Spiegazione per il prof
            st.markdown("### Perché usare K-Means per i Videogiochi?")
            st.write("Il clustering ci aiuta a scoprire le 'ricette' del mercato. Ad esempio, potremmo individuare un cluster di **'Capolavori di nicchia'** (voti alti ma vendite basse) o i classici **'AAA Games'** (voti medi ma incassi stellari). Standardizzando i dati prima di analizzarli, l'algoritmo non viene ingannato dalle diverse scale di grandezza tra anni, voti e milioni di copie vendute.")
elif scelta == "Come Funziona":
    st.header("Dietro le quinte: K-Nearest Neighbors")
    st.write("""
    Questa applicazione utilizza l'algoritmo di Machine Learning **K-Nearest Neighbors (KNN)**.
    
    Immagina che ogni videogioco sia un punto nello spazio. L'algoritmo analizza le caratteristiche di ogni gioco 
    (genere, piattaforma, vendite, recensioni) e calcola la distanza matematica tra questi punti.
    
    Quando selezioni un gioco, il modello cerca i **5 punti più vicini** nello spazio multidimensionale, 
    proponendoti i titoli matematicamente più simili!
    """)

