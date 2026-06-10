import sys
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "btc-price-model"

client = MlflowClient()

try:
with open("latest_run_id.txt", "r") as f:
new_run_id = f.read().strip()

new_run = client.get_run(new_run_id)
new_rmse = new_run.data.metrics.get("rmse")

print(f"New Model RMSE: {new_rmse}")

# Ambil model production saat ini
production_versions = client.get_latest_versions(
    MODEL_NAME,
    stages=["Production"]
)

if not production_versions:
    print("Tidak ada Production model.")
    print("Promote langsung ke Production.")

    model_uri = f"runs:/{new_run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Production"
    )

    sys.exit(0)

production_version = production_versions[0]

prod_run = client.get_run(
    production_version.run_id
)

prod_rmse = prod_run.data.metrics.get("rmse")

print(f"Production RMSE: {prod_rmse}")

# RMSE lebih kecil = lebih bagus
if new_rmse < prod_rmse:

    print("New model BETTER")

    model_uri = f"runs:/{new_run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Production"
    )

    print("Promoted to Production")

else:
    print("New model WORSE")
    print("Keep current Production model")


except Exception as e:
print(e)
sys.exit(1)
