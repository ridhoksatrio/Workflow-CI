import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier

# REVISI: Mengambil kredensial dari environment variable (GitHub Secrets)
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', 'ridhoksatrio')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', 'default_password')

mlflow.set_tracking_uri("https://dagshub.com/ridhoksatrio/Eksperimen_SML_Ridho-Kukuh-Ksatrio.mlflow")

def run_modelling():
    # Pastikan file CSV ini ada di folder yang sama
    df = pd.read_csv("spotify_preprocessing.csv")

    train_df = df[df['set_type'] == 'train'].drop('set_type', axis=1)
    X_train = train_df.drop('pop_category', axis=1)
    y_train = train_df['pop_category']

    mlflow.set_experiment("Spotify_Basic_Modelling")
    
    # Autolog mencatat parameter & metrik otomatis
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Baseline_Model_Final"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_acc)

        # REVISI: Simpan model secara eksplisit agar folder 'model' tercipta di mlruns
        mlflow.sklearn.log_model(model, "spotify-rf-model")

        print(f"Training Accuracy: {train_acc}")

if __name__ == "__main__":
    run_modelling()
