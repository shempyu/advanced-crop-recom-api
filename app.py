import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS

from xgboost import XGBRegressor

# ============================================================
# CREATE APP
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# WEATHER FEATURES
# ============================================================

FEATURES = [
    'pre_sum',
    'qv2m',
    't2m',
    't2m_max',
    't2m_min'
]

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

# ============================================================
# GLOBAL VARIABLES
# ============================================================

weather_models = {}

weather_scaler = None

weather_df = None

basic_crop_model = None

advanced_crop_model = None

basic_crop_scaler = None

advanced_crop_scaler = None

le_crop = None

# ============================================================
# LOAD RESOURCES
# ============================================================

def load_resources():

    global weather_models
    global weather_scaler
    global weather_df

    global basic_crop_model
    global advanced_crop_model

    global basic_crop_scaler
    global advanced_crop_scaler

    global le_crop

    # ========================================================
    # WEATHER
    # ========================================================

    if weather_scaler is None:

        weather_scaler = joblib.load(
            "models/weather_scaler.pkl"
        )

    if weather_df is None:

        weather_df = pd.read_pickle(
            "models/processed_weather_df.pkl"
        )

    if len(weather_models) == 0:

        for feature in FEATURES:

            model = XGBRegressor()

            model.load_model(
                f"models/xgb_{feature}.json"
            )

            weather_models[feature] = model

    # ========================================================
    # CROP MODELS
    # ========================================================

    if basic_crop_model is None:

        basic_crop_model = joblib.load(
            "models/basic_crop_model.pkl"
        )

    if advanced_crop_model is None:

        advanced_crop_model = joblib.load(
            "models/advanced_crop_model.pkl"
        )

    if basic_crop_scaler is None:

        basic_crop_scaler = joblib.load(
            "models/basic_crop_scaler.pkl"
        )

    if advanced_crop_scaler is None:

        advanced_crop_scaler = joblib.load(
            "models/advanced_crop_scaler.pkl"
        )

    if le_crop is None:

        le_crop = joblib.load(
            "models/le_crop.pkl"
        )

# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "message": "Smart Crop AI API Running 🚀"
    })

# ============================================================
# DISTRICTS
# ============================================================

@app.route(
    "/districts",
    methods=["GET"]
)
def get_districts():

    load_resources()

    districts = sorted(
        weather_df['district'].unique()
    )

    return jsonify({
        "districts": districts
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

        load_resources()

        data = request.json

        district = data['district']

        target_year = int(
            data['year']
        )

        target_month = data['month']

        if district not in weather_df['district'].unique():

            return jsonify({
                "error": "Invalid district"
            }), 400

        target_month_num = MONTH_MAP.get(
            target_month
        )

        if not target_month_num:

            return jsonify({
                "error": "Invalid month"
            }), 400

        # ====================================================
        # DISTRICT DATA
        # ====================================================

        d_df = weather_df[
            weather_df['district'] == district
        ]

        history = d_df[
            FEATURES
        ].tail(12).values.tolist()

        last_year = int(
            d_df.iloc[-1]['year']
        )

        last_month = int(
            d_df.iloc[-1]['month_num']
        )

        total_months_needed = (

            (target_year - last_year) * 12

            +

            (target_month_num - last_month)
        )

        if total_months_needed <= 0:

            return jsonify({
                "error": "Date must be future"
            }), 400

        # ====================================================
        # ITERATIVE FORECASTING
        # ====================================================

        for _ in range(total_months_needed):

            input_seq = np.array(
                history[-12:]
            ).flatten().reshape(1, -1)

            pred = []

            for feature in FEATURES:

                value = weather_models[
                    feature
                ].predict(input_seq)[0]

                pred.append(value)

            history.append(pred)

        # ====================================================
        # FINAL PREDICTION
        # ====================================================

        final_scaled = np.array([
            history[-1]
        ])

        final = weather_scaler.inverse_transform(
            final_scaled
        )

        # Reverse rainfall log transform
        final[0,0] = np.expm1(
            final[0,0]
        )

        temperature = float(
            final[0,2]
        )

        humidity = float(
            final[0,1]
        )

        rainfall = max(
            0,
            float(final[0,0])
        )

        return jsonify({

            "temperature": round(
                temperature,
                2
            ),

            "humidity": round(
                humidity,
                2
            ),

            "rainfall": round(
                rainfall,
                2
            ),

            "temp_max": round(
                float(final[0,3]),
                2
            ),

            "temp_min": round(
                float(final[0,4]),
                2
            )
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ============================================================
# BASIC CROP RECOMMENDATION
# ============================================================

@app.route(
    "/recommend-basic-crop",
    methods=["POST"]
)
def recommend_basic_crop():

    try:

        load_resources()

        data = request.json

        input_df = pd.DataFrame(

            [[

                float(data['temperature']),

                float(data['humidity']),

                float(data['rainfall'])

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
            probabilities[top5_indices]
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

        load_resources()

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
            probabilities[top5_indices]
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

    port = int(
        os.environ.get("PORT", 10000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )