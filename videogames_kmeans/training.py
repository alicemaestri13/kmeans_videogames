import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib

print("Inizio addestramento del modello...")

# 1. Carica i dati puliti dal file generato precedentemente
df = pd.read_pickle('dataframe_pulito.pkl')

# 2. Prepara le feature
features = df[['Genre', 'Platform', 'Global_Sales', 'Critic_Score']]
features_encoded = pd.get_dummies(features, columns=['Genre', 'Platform'])

# 3. Normalizza i dati
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_encoded)

# 4. Addestra il modello KNN
knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_model.fit(features_scaled)

# 5. SALVATAGGIO CON JOBLIB
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(features_scaled, 'features_scaled.pkl')

print("Modello e feature salvati con successo su disco!")
