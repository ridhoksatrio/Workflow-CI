import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier

# 1. LOGIKA KREDENSIAL (Sangat Penting untuk CI/CD)
# Mengambil dari GitHub Secrets (untuk CI) atau Environment Variable lokal
token = os.getenv("MLFLOW_TRACKING_PASSWORD")
username = os.getenv("MLFLOW_TRACKING_USERNAME")

if username and token:
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    mlflow.set_tracking_uri("https://dagshub.com/ridhoksatrio/Eksperimen_SML_Ridho-Kukuh-Ksatrio.mlflow")
else:
    # Jika dijalankan lokal dan env tidak ditemukan, baru masukkan manual di sini
    os.environ['MLFLOW_TRACKING_USERNAME'] = "ridhoksatrio"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "867c13746fbdd796fa10a32be6052cf949fb0dcb"
    mlflow.set_tracking_uri("https://dagshub.com/ridhoksatrio/Eksperimen_SML_Ridho-Kukuh-Ksatrio.mlflow")

def run_modelling():
    # 2. PENYESUAIAN PATH DATASET
    # Di MLProject, file csv berada di folder yang sama dengan modelling.py
    try:
        df = pd.read_csv("spotify_preprocessing.csv")
    except FileNotFoundError:
        # Cadangan jika dijalankan dari luar folder
        df = pd.read_csv("MLProject_Folder/spotify_preprocessing.csv")
    
    train_df = df[df['set_type'] == 'train'].drop('set_type', axis=1)
    X_train = train_df.drop('pop_category', axis=1)
    y_train = train_df['pop_category']

    mlflow.set_experiment("Spotify_Basic_Modelling")

    with mlflow.start_run(run_name="Baseline_Model_CI"):
        # Menggunakan autolog untuk kriteria Basic di modelling.py
        mlflow.sklearn.autolog()
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        print("Model Baseline CI berhasil di-log ke DagsHub!")

if __name__ == "__main__":
    run_modelling()