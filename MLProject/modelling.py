import pandas as pd
import mlflow
import mlflow.sklearn
import os
import time  # <--- Perubahan 1: Tambahkan library waktu
from sklearn.ensemble import RandomForestClassifier

# 1. Konfigurasi Kredensial
token = os.getenv("MLFLOW_TRACKING_PASSWORD")
username = os.getenv("MLFLOW_TRACKING_USERNAME")

if username and token:
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    mlflow.set_tracking_uri("https://dagshub.com/ridhoksatrio/Eksperimen_SML_Ridho-Kukuh-Ksatrio.mlflow")
else:
    # Lokal testing
    os.environ['MLFLOW_TRACKING_USERNAME'] = "ridhoksatrio"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "867c13746fbdd796fa10a32be6052cf949fb0dcb"
    mlflow.set_tracking_uri("https://dagshub.com/ridhoksatrio/Eksperimen_SML_Ridho-Kukuh-Ksatrio.mlflow")

def run_modelling():
    # 2. Load Dataset
    # Pastikan file ini ada di folder MLProject bersama modelling.py
    df = pd.read_csv("spotify_preprocessing.csv")
    
    train_df = df[df['set_type'] == 'train'].drop('set_type', axis=1)
    X_train = train_df.drop('pop_category', axis=1)
    y_train = train_df['pop_category']

    # 3. Set Experiment
    mlflow.set_experiment("Spotify_CI_Otomatis")
    
    # <--- Perubahan 2: Jeda waktu agar server DagsHub siap
    print("Menunggu server DagsHub melakukan sinkronisasi eksperimen...")
    time.sleep(5) 

    # 4. Start Run
    try:
        with mlflow.start_run(run_name="Run_via_GitHub_Actions"):
            mlflow.sklearn.autolog()
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Simpan Model sebagai Artefak
            mlflow.sklearn.log_model(model, "model")
            print("Selesai! Model telah terkirim ke DagsHub.")
    except Exception as e:
        print(f"Terjadi kesalahan saat logging ke MLflow: {e}")

if __name__ == "__main__":
    run_modelling()
