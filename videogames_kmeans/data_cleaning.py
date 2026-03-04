import pandas as pd

print("Inizio pulizia dati...")

# Carica il file originale scaricato da Kaggle
df = pd.read_csv('Video_Games.csv')

# Filtra le colonne che ci servono
df = df[['Name', 'Year_of_Release', 'Genre', 'Platform', 'Global_Sales', 'Critic_Score']]

# Rimuove TUTTE le righe che hanno almeno un valore mancante (NaN)
df = df.dropna()

# Crea la colonna Decade
df['Decade'] = (df['Year_of_Release'] // 10) * 10

# Resetta gli indici (molto importante per evitare che il KNN si confonda dopo aver rimosso i NaN)
df = df.reset_index(drop=True)

# Salva il dataframe pulito in un file .pkl (pickle)
df.to_pickle('dataframe_pulito.pkl')

print("Dati puliti e salvati in 'dataframe_pulito.pkl'!")
