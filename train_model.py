import warnings
warnings.filterwarnings('ignore')

import os
import joblib
import numpy as np
import pandas as pd

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

from xgboost import XGBRegressor

# ============================================================
# CREATE MODELS DIRECTORY
# ============================================================

os.makedirs("models", exist_ok=True)

# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...\n")

weather_df = pd.read_csv(
    "DATA_SET2.csv",
    skiprows=5,
    header=None
)

crop_df = pd.read_csv(
    "croprecommendationml.csv"
)

# ============================================================
# WEATHER DATA CLEANING
# ============================================================

weather_df = weather_df.iloc[:, :8]

weather_df.columns = [
    'district',
    'year',
    'month',
    'pre_sum',
    'qv2m',
    't2m',
    't2m_max',
    't2m_min'
]

weather_df = weather_df.replace('ANN', np.nan)

weather_df = weather_df[
    weather_df['district'] != 'District'
]

weather_df['district'] = (
    weather_df['district']
    .astype(str)
    .str.strip()
)

weather_df['month'] = (
    weather_df['month']
    .astype(str)
    .str.strip()
)

FEATURES = [
    'pre_sum',
    'qv2m',
    't2m',
    't2m_max',
    't2m_min'
]

for col in FEATURES:

    weather_df[col] = pd.to_numeric(
        weather_df[col],
        errors='coerce'
    )

weather_df['year'] = pd.to_numeric(
    weather_df['year'],
    errors='coerce'
)

weather_df = weather_df.dropna().reset_index(drop=True)

# ============================================================
# LOG TRANSFORM RAINFALL
# ============================================================

weather_df['pre_sum'] = np.log1p(
    weather_df['pre_sum']
)

# ============================================================
# MONTH NUMBER
# ============================================================

month_map = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12
}

weather_df['month_num'] = (
    weather_df['month'].map(month_map)
)

# ============================================================
# SORT DATA
# ============================================================

weather_df = weather_df.sort_values(
    ['district', 'year', 'month_num']
).reset_index(drop=True)

print("Weather Data Shape :", weather_df.shape)

# ============================================================
# WEATHER SCALER
# ============================================================

weather_scaler = MinMaxScaler()

weather_df[FEATURES] = (
    weather_scaler.fit_transform(
        weather_df[FEATURES]
    )
)

# Save scaler
joblib.dump(
    weather_scaler,
    "models/weather_scaler.pkl"
)

# Save processed dataframe
weather_df.to_pickle(
    "models/processed_weather_df.pkl"
)

# ============================================================
# CREATE WEATHER SEQUENCES
# ============================================================

def create_weather_sequences(
    data,
    input_steps=12
):

    X = []
    y = []

    for district in data['district'].unique():

        d_data = data[
            data['district'] == district
        ][FEATURES].values

        if len(d_data) <= input_steps:
            continue

        for i in range(
            len(d_data) - input_steps
        ):

            seq_x = d_data[
                i:i+input_steps
            ].flatten()

            seq_y = d_data[
                i+input_steps
            ]

            X.append(seq_x)
            y.append(seq_y)

    return np.array(X), np.array(y)

X_weather, y_weather = (
    create_weather_sequences(
        weather_df
    )
)

print("\nWeather X Shape :", X_weather.shape)
print("Weather y Shape :", y_weather.shape)

# ============================================================
# TRAIN XGBOOST WEATHER MODELS
# ============================================================

print("\nTraining XGBoost Weather Models...\n")

weather_models = {}

for i, feature in enumerate(FEATURES):

    print(f"Training {feature} model...")

    model = XGBRegressor(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=6,

        subsample=0.8,

        colsample_bytree=0.8,

        objective='reg:squarederror',

        random_state=42,

        n_jobs=-1
    )

    model.fit(
        X_weather,
        y_weather[:, i]
    )

    weather_models[feature] = model

    model.save_model(
        f"models/xgb_{feature}.json"
    )

    print(f"Saved xgb_{feature}.json")

# ============================================================
# CROP DATA CLEANING
# ============================================================

crop_df.columns = (
    crop_df.columns
    .str.strip()
    .str.lower()
)

le_crop = LabelEncoder()

crop_df['label_enc'] = (
    le_crop.fit_transform(
        crop_df['label']
    )
)

# ============================================================
# BASIC CROP FEATURES
# ============================================================

basic_crop_scaler = MinMaxScaler()

X_crop_basic = crop_df[
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

advanced_crop_scaler = MinMaxScaler()

X_crop_advanced = crop_df[
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

y_crop = crop_df['label_enc']

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
# SAVE CROP MODELS
# ============================================================

joblib.dump(
    basic_crop_model,
    "models/basic_crop_model.pkl"
)

joblib.dump(
    advanced_crop_model,
    "models/advanced_crop_model.pkl"
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
    le_crop,
    "models/le_crop.pkl"
)

print("\n✅ ALL MODELS SAVED SUCCESSFULLY!")