import pandas as pd
import joblib

print("Caricamento modello in corso...\n")

# 1. Carica modello e dati salvati
knn_model = joblib.load('knn_model.pkl')
df = pd.read_pickle('dataframe_pulito.pkl')
features_scaled = joblib.load('features_scaled.pkl')

# 2. Definisce la funzione di raccomandazione
def raccomanda_giochi(nome_gioco, modello, df_originale, dati_processati):
    idx_list = df_originale[df_originale['Name'] == nome_gioco].index
    
    if len(idx_list) == 0:
        return f"Mi dispiace, '{nome_gioco}' non è nel database o è stato scartato durante la pulizia."
    
    idx = idx_list[0]
    
    # Trova i vicini
    distanze, indici = modello.kneighbors([dati_processati[idx]])
    
    print(f"🎮 Giochi consigliati se ti piace '{nome_gioco}':\n")
    for i in range(1, len(indici[0])):
        indice_vicino = indici[0][i]
        nome_vicino = df_originale.iloc[indice_vicino]['Name']
        piattaforma = df_originale.iloc[indice_vicino]['Platform']
        genere = df_originale.iloc[indice_vicino]['Genre']
        print(f"{i}. {nome_vicino} ({piattaforma} - {genere})")

# 3. Testa il motore di ricerca!
raccomanda_giochi("Grand Theft Auto V", knn_model, df, features_scaled)
