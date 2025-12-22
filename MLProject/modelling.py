import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier

# 1. Konfigurasi Kredensial (Gunakan Secrets agar Aman)
# Ini akan mengambil nilai dari GitHub Secrets yang Anda isi tadi
token = os.getenv("MLFLOW_TRACKING_PASSWORD")
username = os.getenv("MLFLOW_TRACKING_USERNAME")

if username and token:
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    # Pastikan URL ini sesuai dengan repo DagsHub Anda
    mlflow.set_tracking_uri("https://dagshub.com/ridhoksatrio/Eksperimen_SML_Ridho-Kukuh-Ksatrio.mlflow")
else:
    # Jika dijalankan lokal tanpa env, gunakan manual (Opsional)
    os.environ['MLFLOW_TRACKING_USERNAME'] = "ridhoksatrio"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "TOKEN_DAGSHUB_ANDA"
    mlflow.set_tracking_uri("https://dagshub.com/ridhoksatrio/Eksperimen_SML_Ridho-Kukuh-Ksatrio.mlflow")

def run_modelling():
    # 2. Load Dataset (Pastikan file CSV ada di folder yang sama)
    df = pd.read_csv("spotify_preprocessing.csv")
    
    train_df = df[df['set_type'] == 'train'].drop('set_type', axis=1)
    X_train = train_df.drop('pop_category', axis=1)
    y_train = train_df['pop_category']

    mlflow.set_experiment("Spotify_CI_Otomatis")

    with mlflow.start_run(run_name="Run_via_GitHub_Actions"):
        # Kriteria Basic/Skilled minta autolog atau manual logging
        mlflow.sklearn.autolog()
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan Model sebagai Artefak
        mlflow.sklearn.log_model(model, "model")
        print("Selesai! Model telah terkirim ke DagsHub.")

if __name__ == "__main__":
    run_modelling()
