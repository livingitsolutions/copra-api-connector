from fastapi import FastAPI
import threading
import time
import requests

app = FastAPI()

# CONFIG
LARAVEL_API = "https://ict.ccsit.info/api"
PREDICT_API = "https://copra-fastapi2.onrender.com/predict"

# STATE
current_batch_id = None
is_running = False
latest_sensor_data = None


# ----------------------------
# 📦 MODELS
# ----------------------------
class StartRequest(BaseModel):
    batch_id: int

class SensorData(BaseModel):
    moisture: float
    temperature: float
    color: int


# ----------------------------
# ▶ START TEST
# ----------------------------
@app.post("/start-test")
def start_test(data: StartRequest):
    global current_batch_id, is_running

    if is_running:
        return {"status": "already running"}

    current_batch_id = data.batch_id
    is_running = True

    thread = threading.Thread(target=run_test_loop)
    thread.start()

    return {
        "status": "started",
        "batch_id": current_batch_id
    }


# ----------------------------
# ⏹ STOP TEST
# ----------------------------
@app.post("/stop-test")
def stop_test(data: StartRequest):
    global is_running, current_batch_id

    is_running = False
    batch_id = current_batch_id
    current_batch_id = None

    return {
        "status": "stopped",
        "batch_id": batch_id
    }


# ----------------------------
# 📡 SENSOR INPUT
# ----------------------------
@app.post("/sensor-data")
def receive_sensor_data(data: SensorData):
    global latest_sensor_data

    latest_sensor_data = data.dict()

    return {
        "status": "received"
    }


# ----------------------------
# 🔁 MAIN LOOP
# ----------------------------
def run_test_loop():
    global is_running, current_batch_id, latest_sensor_data

    while is_running and current_batch_id:
        try:
            if not latest_sensor_data:
                time.sleep(1)
                continue

            sensor_data = latest_sensor_data

            # CALL ML MODEL
            predict_res = requests.get(PREDICT_API, params=sensor_data)

            if predict_res.status_code != 200:
                print("Prediction failed")
                continue

            prediction = predict_res.json()

            # SEND TO LARAVEL
            payload = {
                "moisture": sensor_data["moisture"],
                "temperature": sensor_data["temperature"],
                "svm_grade": prediction.get("SVM"),
                "rf_grade": prediction.get("Random Forest"),
                "knn_grade": prediction.get("KNN"),
                "lr_grade": prediction.get("Logistic Regression"),
            }

            requests.post(
                f"{LARAVEL_API}/batch/{current_batch_id}/samples",
                json=payload
            )

            print(f"Sample sent (batch {current_batch_id})")

            # prevent duplicates
            latest_sensor_data = None

        except Exception as e:
            print("Error:", str(e))

        time.sleep(1)