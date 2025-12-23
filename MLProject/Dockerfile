# Gunakan image Python yang ringan
FROM python:3.12-slim

# Set folder kerja di dalam container
WORKDIR /app

# Salin semua file proyek ke dalam container
# Pastikan saat build, konteksnya adalah folder MLProject
COPY . /app

# Install library yang dibutuhkan
RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn pandas numpy dagshub

CMD ["mlflow", "run", ".", "-e", "main", "--env-manager", "local"]
