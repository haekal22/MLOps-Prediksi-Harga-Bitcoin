import sys
import mlflow
from mlflow.tracking import MlflowClient

THRESHOLD = 800.0
MODEL_NAME = "btc-price-model"

# Connect ke MLflow (Bisa dilewatkan via ENV variabel di Github Actions)
client = MlflowClient()

try:
    # Ambil eksperimen
    experiment = client.get_experiment_by_name("BTC Prediction")
    if not experiment:
        print("❌ Eksperimen 'BTC Prediction' tidak ditemukan.")
        sys.exit(1)

    # Cari run terakhir yang baru saja diselesaikan oleh job 'train'
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=["attributes.start_time DESC"]
    )

    if not runs:
        print("❌ Tidak ada riwayat run untuk dievaluasi.")
        sys.exit(1)

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    rmse = latest_run.data.metrics.get("rmse", float("inf"))

    print(f"📊 Mengevaluasi Run ID: {run_id}")
    print(f"📊 Nilai Real RMSE: {rmse} (Ambang Batas Threshold: {THRESHOLD})")

    if rmse < THRESHOLD:
        print("✅ Model VALID: Performa memenuhi syarat standar.")
        # Simpan Run ID ke file sementara agar job 'registry' bisa membacanya nanti
        with open("latest_run_id.txt", "w") as f:
            f.write(run_id)
        sys.exit(0)
    else:
        print("❌ Model FAILED: Performa buruk, di bawah standar.")
        sys.exit(1)

except Exception as e:
    print(f"❌ Error saat melakukan evaluasi: {e}")
    sys.exit(1)