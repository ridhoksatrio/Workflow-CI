import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier

# Mengambil kredensial dari environment variable (GitHub Secrets)
# Gunakan os.getenv agar fleksibel antara lokal dan CI/CD
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', 'ridhoksatrio')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', '867c13746fbdd796fa10a32be6052cf949fb0dcb')

mlflow.set_tracking_uri("https://dagshub.com/ridhoksatrio/Eksperimen_SML_Ridho-Kukuh-Ksatrio.mlflow")

def run_modelling():
    # Membaca data - Pastikan file ini ada di folder MLProject
    df = pd.read_csv("spotify_preprocessing.csv")

    train_df = df[df['set_type'] == 'train'].drop('set_type', axis=1)
    X_train = train_df.drop('pop_category', axis=1)
    y_train = train_df['pop_category']

    mlflow.set_experiment("Spotify_Basic_Modelling")
    
    # Autolog mencatat parameter & metrik secara otomatis
    mlflow.sklearn.autolog()

    # REVISI PENTING: Tambahkan nested=True untuk menghindari error "Run not found" di CI/CD
    with mlflow.start_run(run_name="Baseline_Model_Final", nested=True):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_acc)

        # Menyimpan model secara eksplisit ke dalam artefak MLflow
        mlflow.sklearn.log_model(model, "spotify-rf-model")

        print(f"Model Baseline berhasil di-log!")
        print(f"Training Accuracy: {train_acc}")

if __name__ == "__main__":
    run_modelling()
