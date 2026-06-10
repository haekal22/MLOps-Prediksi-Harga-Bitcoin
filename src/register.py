import sys
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "btc-price-model"
client = MlflowClient()

try:
    # Membaca Run ID yang divalidasi oleh tahap evaluate tadi
    with open("latest_run_id.txt", "r") as f:
        run_id = f.read().strip()

    model_uri = f"runs:/{run_id}/model"
    print(f"📦 Mendaftarkan Model dari Run ID: {run_id} ke Model Registry...")

    # Daftarkan ke MLflow Registry
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    # Ubah status versi terbaru tersebut menjadi 'Staging'

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Staging",
        archive_existing_versions=True
    )

    print(f"🎉 SUCCESS: Model '{MODEL_NAME}' Versi {result.version} resmi berstatus STAGING!")

except FileNotFoundError:
    print("❌ File 'latest_run_id.txt' tidak ditemukan. Tahap registrasi gagal.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Gagal mendaftarkan model ke registry: {e}")
    sys.exit(1)