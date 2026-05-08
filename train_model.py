# ============================================================
# SMART CROP AI - TRAINING SCRIPT
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

from sklearn.preprocessing import (
    MinMaxScaler,
    LabelEncoder
)

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Conv1D,
    BatchNormalization,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    Add
)

from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# CREATE MODELS DIRECTORY
# ============================================================

os.makedirs("models", exist_ok=True)

# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...\n")

weather_data = pd.read_csv(
    "DATA SET2 (1).csv",
    skiprows=5
)

crop_data = pd.read_csv(
    "croprecommendationml.csv"
)

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

weather_data.columns = (
    weather_data.columns
    .str.strip()
    .str.lower()
)

crop_data.columns = (
    crop_data.columns
    .str.strip()
    .str.lower()
)

# ============================================================
# CLEAN WEATHER DATA
# ============================================================

weather_data['district'] = (
    weather_data['district']
    .astype(str)
    .str.strip()
    .str.lower()
)

weather_data['month'] = (
    weather_data['month']
    .astype(str)
    .str.strip()
    .str.lower()
)

# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_cols = [
    'year',
    'pre_sum',
    'qv2m',
    't2m',
    't2m_max',
    't2m_min'
]

for col in numeric_cols:

    weather_data[col] = pd.to_numeric(
        weather_data[col],
        errors='coerce'
    )

weather_data.dropna(inplace=True)

# ============================================================
# LOG TRANSFORM RAINFALL
# ============================================================

weather_data['pre_sum'] = np.log1p(
    weather_data['pre_sum']
)

# ============================================================
# HUMIDITY CONVERSION
# ============================================================

def convert_qv2m_to_rh(
    q,
    temp_c,
    pressure=1013.25
):

    if q > 1:
        q = q / 1000

    q = max(q, 1e-6)

    e = (
        q * pressure
    ) / (
        0.622 + 0.378 * q
    )

    es = 6.112 * np.exp(
        (17.67 * temp_c) /
        (temp_c + 243.5)
    )

    rh = (e / es) * 100

    return np.clip(rh, 10, 100)

weather_data['humidity'] = weather_data.apply(

    lambda row: convert_qv2m_to_rh(
        row['qv2m'],
        row['t2m']
    ),

    axis=1
)

# ============================================================
# LABEL ENCODING
# ============================================================

le_district = LabelEncoder()
le_month = LabelEncoder()
le_crop = LabelEncoder()

weather_data['district_enc'] = (
    le_district.fit_transform(
        weather_data['district']
    )
)

weather_data['month_enc'] = (
    le_month.fit_transform(
        weather_data['month']
    )
)

crop_data['label_enc'] = (
    le_crop.fit_transform(
        crop_data['label']
    )
)

# ============================================================
# FEATURE SCALERS
# ============================================================

climate_scaler = MinMaxScaler()

basic_crop_scaler = MinMaxScaler()

advanced_crop_scaler = MinMaxScaler()

# ============================================================
# WEATHER FEATURES
# ============================================================

X_climate = weather_data[
    ['district_enc', 'month_enc', 'year']
]

X_climate_scaled = climate_scaler.fit_transform(
    X_climate
)

# ============================================================
# BASIC CROP FEATURES
# ============================================================

X_crop_basic = crop_data[
    [
        'temperature',
        'humidity',
        'rainfall'
    ]
]

X_crop_basic_scaled = (
    basic_crop_scaler.fit_transform(
        X_crop_basic
    )
)

# ============================================================
# ADVANCED CROP FEATURES
# ============================================================

X_crop_advanced = crop_data[
    [
        'temperature',
        'humidity',
        'rainfall',
        'n',
        'p',
        'k',
        'ph'
    ]
]

X_crop_advanced_scaled = (
    advanced_crop_scaler.fit_transform(
        X_crop_advanced
    )
)

# ============================================================
# RESHAPE FOR DL
# ============================================================

X_climate_lstm = X_climate_scaled.reshape(
    X_climate_scaled.shape[0],
    X_climate_scaled.shape[1],
    1
)

# ============================================================
# TARGETS
# ============================================================

y_temp = weather_data['t2m'].values

y_hum = weather_data['humidity'].values

y_rain = weather_data['pre_sum'].values

y_crop = crop_data['label_enc']

# ============================================================
# WEATHER MODEL
# ============================================================

def build_weather_model():

    inputs = Input(
        shape=(
            X_climate_lstm.shape[1],
            1
        )
    )

    x1 = Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    )(inputs)

    x1 = Dropout(0.2)(x1)

    x2 = LSTM(
        64,
        return_sequences=True
    )(x1)

    x2 = Dropout(0.2)(x2)

    x3 = Conv1D(
        filters=64,
        kernel_size=3,
        padding='causal',
        activation='relu'
    )(x2)

    x3 = BatchNormalization()(x3)

    x4 = Add()([x2, x3])

    x5 = GlobalAveragePooling1D()(x4)

    x5 = Dense(
        128,
        activation='relu'
    )(x5)

    x5 = Dropout(0.3)(x5)

    x5 = Dense(
        64,
        activation='relu'
    )(x5)

    outputs = Dense(1)(x5)

    model = Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer='adam',
        loss='huber',
        metrics=['mae']
    )

    return model

# ============================================================
# TRAIN WEATHER MODELS
# ============================================================

print("\nTraining Weather Models...\n")

temp_model = build_weather_model()
hum_model = build_weather_model()
rain_model = build_weather_model()

early_stop = EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=True
)

temp_model.fit(
    X_climate_lstm,
    y_temp,
    epochs=20,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]
)

hum_model.fit(
    X_climate_lstm,
    y_hum,
    epochs=20,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]
)

rain_model.fit(
    X_climate_lstm,
    y_rain,
    epochs=20,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]
)

# ============================================================
# BASIC CROP MODEL
# ============================================================

print("\nTraining Basic Crop Model...\n")

basic_crop_model = VotingClassifier(

    estimators=[

        (
            'dt',
            DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            )
        ),

        (
            'nb',
            GaussianNB()
        ),

        (
            'rf',
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )

    ],

    voting='soft'
)

basic_crop_model.fit(
    X_crop_basic_scaled,
    y_crop
)

# ============================================================
# ADVANCED CROP MODEL
# ============================================================

print("\nTraining Advanced Crop Model...\n")

advanced_crop_model = VotingClassifier(

    estimators=[

        (
            'dt',
            DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            )
        ),

        (
            'nb',
            GaussianNB()
        ),

        (
            'rf',
            RandomForestClassifier(
                n_estimators=300,
                random_state=42
            )
        )

    ],

    voting='soft'
)

advanced_crop_model.fit(
    X_crop_advanced_scaled,
    y_crop
)

# ============================================================
# SAVE MODELS
# ============================================================

print("\nSaving models...\n")

temp_model.save(
    "models/temp_model.keras"
)

hum_model.save(
    "models/hum_model.keras"
)

rain_model.save(
    "models/rain_model.keras"
)

joblib.dump(
    basic_crop_model,
    "models/basic_crop_model.pkl"
)

joblib.dump(
    advanced_crop_model,
    "models/advanced_crop_model.pkl"
)

joblib.dump(
    climate_scaler,
    "models/climate_scaler.pkl"
)

joblib.dump(
    basic_crop_scaler,
    "models/basic_crop_scaler.pkl"
)

joblib.dump(
    advanced_crop_scaler,
    "models/advanced_crop_scaler.pkl"
)

joblib.dump(
    le_district,
    "models/le_district.pkl"
)

joblib.dump(
    le_month,
    "models/le_month.pkl"
)

joblib.dump(
    le_crop,
    "models/le_crop.pkl"
)

print("\nAll models saved successfully!")