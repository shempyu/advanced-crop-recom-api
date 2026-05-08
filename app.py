from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# ============================================================
# CREATE APP
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

temp_model = tf.keras.models.load_model(
    "models/temp_model.keras",
    compile=False
)

hum_model = tf.keras.models.load_model(
    "models/hum_model.keras",
    compile=False
)

rain_model = tf.keras.models.load_model(
    "models/rain_model.keras",
    compile=False
)

basic_crop_model = joblib.load(
    "models/basic_crop_model.pkl"
)

advanced_crop_model = joblib.load(
    "models/advanced_crop_model.pkl"
)

climate_scaler = joblib.load(
    "models/climate_scaler.pkl"
)

basic_crop_scaler = joblib.load(
    "models/basic_crop_scaler.pkl"
)

advanced_crop_scaler = joblib.load(
    "models/advanced_crop_scaler.pkl"
)

le_district = joblib.load(
    "models/le_district.pkl"
)

le_month = joblib.load(
    "models/le_month.pkl"
)

le_crop = joblib.load(
    "models/le_crop.pkl"
)

print("Models loaded successfully!")

# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "message": "Smart Crop API Running!"
    })

# ============================================================
# WEATHER PREDICTION
# ============================================================

@app.route(
    "/predict-weather",
    methods=["POST"]
)
def predict_weather():

    try:

        data = request.json

        district = data['district'].lower()
        year = int(data['year'])
        month = data['month'].lower()

        district_enc = (
            le_district.transform(
                [district]
            )[0]
        )

        month_enc = (
            le_month.transform(
                [month]
            )[0]
        )

        climate_input = pd.DataFrame(

            [[
                district_enc,
                month_enc,
                year
            ]],

            columns=[
                'district_enc',
                'month_enc',
                'year'
            ]
        )

        climate_scaled = (
            climate_scaler.transform(
                climate_input
            )
        )

        climate_lstm = climate_scaled.reshape(
            1,
            climate_scaled.shape[1],
            1
        )

        pred_temp = (
            temp_model.predict(
                climate_lstm,
                verbose=0
            )[0][0]
        )

        pred_hum = (
            hum_model.predict(
                climate_lstm,
                verbose=0
            )[0][0]
        )

        pred_rain_log = (
            rain_model.predict(
                climate_lstm,
                verbose=0
            )[0][0]
        )

        pred_rain = np.expm1(
            pred_rain_log
        )

        return jsonify({

            "temperature": round(
                float(pred_temp),
                2
            ),

            "humidity": round(
                float(pred_hum),
                2
            ),

            "rainfall": round(
                float(pred_rain),
                2
            )

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# ============================================================
# BASIC CROP RECOMMENDATION
# ============================================================

@app.route(
    "/recommend-basic-crop",
    methods=["POST"]
)
def recommend_basic_crop():

    try:

        data = request.json

        temperature = float(
            data['temperature']
        )

        humidity = float(
            data['humidity']
        )

        rainfall = float(
            data['rainfall']
        )

        input_df = pd.DataFrame(

            [[
                temperature,
                humidity,
                rainfall
            ]],

            columns=[
                'temperature',
                'humidity',
                'rainfall'
            ]
        )

        scaled = (
            basic_crop_scaler.transform(
                input_df
            )
        )

        probabilities = (
            basic_crop_model.predict_proba(
                scaled
            )[0]
        )

        top5_indices = np.argsort(
            probabilities
        )[::-1][:5]

        top5_crops = (
            le_crop.inverse_transform(
                top5_indices
            )
        )

        selected_probs = (
            probabilities[
                top5_indices
            ]
        )

        normalized_scores = (
            selected_probs /
            selected_probs.sum()
        ) * 100

        results = []

        for crop, score in zip(
            top5_crops,
            normalized_scores
        ):

            results.append({

                "crop": crop,

                "confidence": round(
                    float(score),
                    2
                )

            })

        return jsonify({
            "recommended_crops": results
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# ============================================================
# ADVANCED CROP RECOMMENDATION
# ============================================================

@app.route(
    "/recommend-advanced-crop",
    methods=["POST"]
)
def recommend_advanced_crop():

    try:

        data = request.json

        input_df = pd.DataFrame(

            [[
                float(data['temperature']),
                float(data['humidity']),
                float(data['rainfall']),
                float(data['n']),
                float(data['p']),
                float(data['k']),
                float(data['ph'])
            ]],

            columns=[
                'temperature',
                'humidity',
                'rainfall',
                'n',
                'p',
                'k',
                'ph'
            ]
        )

        scaled = (
            advanced_crop_scaler.transform(
                input_df
            )
        )

        probabilities = (
            advanced_crop_model.predict_proba(
                scaled
            )[0]
        )

        top5_indices = np.argsort(
            probabilities
        )[::-1][:5]

        top5_crops = (
            le_crop.inverse_transform(
                top5_indices
            )
        )

        selected_probs = (
            probabilities[
                top5_indices
            ]
        )

        normalized_scores = (
            selected_probs /
            selected_probs.sum()
        ) * 100

        results = []

        for crop, score in zip(
            top5_crops,
            normalized_scores
        ):

            results.append({

                "crop": crop,

                "confidence": round(
                    float(score),
                    2
                )

            })

        return jsonify({
            "recommended_crops": results
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=10000
    )


