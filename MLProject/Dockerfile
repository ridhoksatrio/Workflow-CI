# Gunakan image Python yang ringan
FROM python:3.12-slim

# Set folder kerja di dalam container
WORKDIR /app

# Salin semua file dari folder MLProject ke dalam container
COPY . /app

# Install library yang dibutuhkan
RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn pandas numpy dagshub

# Perintah untuk menjalankan model saat container dinyalakan
CMD ["python", "modelling.py"]
