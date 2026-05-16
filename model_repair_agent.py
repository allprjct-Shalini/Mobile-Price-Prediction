import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = "Cellphone.csv"
FEATURES = [
    "battery",
    "ram",
    "ppi",
    "resoloution",
    "weight",
    "internal mem",
    "cpu freq",
    "Front_Cam",
    "RearCam",
]
TARGET = "Price"


def train_and_save_model():
    df = pd.read_csv(DATA_FILE)
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    return model, scaler


def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    except Exception:
        return train_and_save_model()


if __name__ == "__main__":
    model, scaler = load_model_and_scaler()
    print("Model type:", type(model))
    print("Scaler type:", type(scaler))
