import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier

# Setup Kredensial dari GitHub Secrets
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', 'ridhoksatrio')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', '867c13746fbdd796fa10a32be6052cf949fb0dcb')

mlflow.set_tracking_uri("https://dagshub.com/ridhoksatrio/Eksperimen_SML_Ridho-Kukuh-Ksatrio.mlflow")

def run_modelling():
    # Membaca data
    df = pd.read_csv("spotify_preprocessing.csv")

    train_df = df[df['set_type'] == 'train'].drop('set_type', axis=1)
    X_train = train_df.drop('pop_category', axis=1)
    y_train = train_df['pop_category']

    # Set eksperimen
    mlflow.set_experiment("Spotify_Basic_Modelling")
    
    # Autolog akan otomatis mencatat parameter & metrik saat model.fit()
    mlflow.sklearn.autolog()

    # --- REVISI: LANGSUNG TRAINING TANPA start_run ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    
    # Mencatat metrik tambahan dan menyimpan model fisik
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.sklearn.log_model(model, "spotify-rf-model")

    print(f"Berhasil! Training Accuracy: {train_acc}")

if __name__ == "__main__":
    run_modelling()
