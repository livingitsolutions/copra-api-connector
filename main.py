# from fastapi import FastAPI
# from pydantic import BaseModel
# import requests
# import threading
# import time

# app = FastAPI()

# # CONFIG
# LARAVEL_API = "https://ict.ccsit.info/api"
# PREDICT_API = "https://copra-fastapi2.onrender.com/predict"

# # STATE
# current_batch_id = None
# is_running = False
# latest_sensor_data = None


# # ----------------------------
# # 📦 MODELS
# # ----------------------------
# class StartRequest(BaseModel):
#     batch_id: int

# class SensorData(BaseModel):
#     moisture: float
#     temperature: float
#     rgb: int


# # ----------------------------
# # ▶ START TEST
# # ----------------------------
# @app.post("/start-test")
# def start_test(data: StartRequest):
#     global current_batch_id, is_running 

#     if is_running:
#         return {"status": "already running"}

#     current_batch_id = data.batch_id
#     is_running = True

#     thread = threading.Thread(target=run_test_loop)
#     thread.start()

#     return {
#         "status": "started",
#         "batch_id": current_batch_id
#     }


# # ----------------------------
# # ⏹ STOP TEST
# # ----------------------------
# @app.post("/stop-test")
# def stop_test(data: StartRequest):
#     global is_running, current_batch_id

#     is_running = False
#     batch_id = current_batch_id
#     current_batch_id = None

#     return {
#         "status": "stopped",
#         "batch_id": batch_id
#     }


# # ----------------------------
# # 📡 SENSOR INPUT
# # ----------------------------
# @app.post("/sensor-data")
# def receive_sensor_data(data: SensorData):
#     global latest_sensor_data

#     latest_sensor_data = data.dict()

#     return {
#         "status": "received"
#     }


# # ----------------------------
# # 🔁 MAIN LOOP
# # ----------------------------
# def run_test_loop():
#     global is_running, current_batch_id, latest_sensor_data

#     while is_running and current_batch_id:
#         try:
#             if not latest_sensor_data:
#                 time.sleep(1)
#                 continue

#             sensor_data = latest_sensor_data
#             # CALL ML MODEL
#             predict_res = requests.post(PREDICT_API,json={
#                 "moisture": sensor_data["moisture"],
#                 "temperature": sensor_data["temperature"],
#                 "color": sensor_data["rgb"]
#             })

#             if predict_res.status_code != 200:
#                 print("Prediction failed")
#                 continue

#             prediction = predict_res.json()

#             # SEND TO LARAVEL
#             payload = {
#                 "moisture": sensor_data["moisture"],
#                 "temperature": sensor_data["temperature"],
#                 "svm_grade": prediction.get("SVM"),
#                 "rf_grade": prediction.get("Random Forest"),
#                 "knn_grade": prediction.get("KNN"),
#                 "lr_grade": prediction.get("Logistic Regression"),
#             }

#             print(f"Payload to be sent ({payload})")

#             requests.post(
#                 f"{LARAVEL_API}/batch/{current_batch_id}/samples",
#                 json=payload
#             )

#             print(f"Sample sent (batch {current_batch_id})")

#             # prevent duplicates
#             latest_sensor_data = None

#         except Exception as e:
#             print("Error:", str(e))

#         time.sleep(1)


from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import httpx

app = FastAPI()

# CONFIG
LARAVEL_API = "https://ict.ccsit.info/api"
PREDICT_API = "https://copra-fastapi3.onrender.com/predict"

# STATE
current_batch_id = None
is_running = False

# Async queue (replaces latest_sensor_data)
sensor_queue = asyncio.Queue()

# ----------------------------
# 📦 MODELS
# ----------------------------
class StartRequest(BaseModel):
    batch_id: int

class SensorData(BaseModel):
    r: int
    g: int
    b: int
    temperature: float
    moisture: float


# ----------------------------
# 🚀 BACKGROUND LOOP (ASYNC)
# ----------------------------
async def run_test_loop():
    global is_running, current_batch_id

    async with httpx.AsyncClient(timeout=60) as client:
        while is_running and current_batch_id is not None:
            try:
                # Wait for sensor data (no polling!)
                sensor_data = await sensor_queue.get()
                if not all(k in sensor_data for k in ("r", "g", "b")):
                    print("⚠️ Skipping incomplete data:", sensor_data)
                    continue
                print(f"Sensor data: {sensor_data}")

                # 🔮 CALL ML API
                predict_res = await client.post(
                    PREDICT_API,
                    json={
                        "moisture": sensor_data["moisture"],
                        "temperature": sensor_data["temperature"],
                        "r": sensor_data["r"],
                        "g": sensor_data["g"],
                        "b": sensor_data["b"],
                    }
                )

                if predict_res.status_code != 200:
                    print("Prediction failed:", predict_res.text)
                    continue

                prediction = predict_res.json()
                predictions_in = prediction.get("input", {})
                predictions_out = prediction.get("predictions", {})
                

                print(f"Prediction data: {predictions_out}")
                

                # 📦 Prepare payload
                payload = {
                    "moisture": predictions_in["Moisture"],
                    "temperature": predictions_in["Temperature"],
                    "r": sensor_data["r"],
                    "g": sensor_data["g"],
                    "b": sensor_data["b"],
                    "svm_grade": predictions_out.get("SVM"),
                    "rf_grade": predictions_out.get("Random Forest"),
                    "knn_grade": predictions_out.get("KNN"),
                    "lr_grade": predictions_out.get("Logistic Regression"),
                }

                print("Sending:", payload)

                # 📡 SEND TO LARAVEL
                laravel_res = await client.post(
                    f"{LARAVEL_API}/batch/{current_batch_id}/samples",
                    json=payload
                )

                if laravel_res.status_code != 200:
                    print("Laravel failed:", laravel_res.text)

            except Exception as e:
                print("Error:", str(e))


# ----------------------------
# ▶ START TEST
# ----------------------------
@app.post("/start-test")
async def start_test(data: StartRequest):
    global current_batch_id, is_running

    if is_running:
        return {"status": "already running"}

    current_batch_id = data.batch_id
    is_running = True

    # Start async background task
    asyncio.create_task(run_test_loop())

    return {
        "status": "started",
        "batch_id": current_batch_id
    }


# ----------------------------
# ⏹ STOP TEST
# ----------------------------
@app.post("/stop-test")
async def stop_test():
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
async def receive_sensor_data(data: SensorData):
    await sensor_queue.put(data.dict())

    return {"status": "queued"}
