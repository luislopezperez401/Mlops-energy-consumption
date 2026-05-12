"""
⚡ Energy Consumption Predictor — PJME Hourly
=============================================
Entrena el mejor modelo (LightGBM) sobre el dataset PJM Hourly Energy Consumption
y levanta una API FastAPI lista para producción.

Uso:
    # 1. Instalar dependencias
    pip install fastapi uvicorn lightgbm scikit-learn pandas numpy joblib

    # 2. Entrenar y guardar el modelo
    python energy_api.py train --data /ruta/a/PJME_hourly.csv

    # 3. Levantar la API
    python energy_api.py serve

    # 4. Documentación interactiva
    http://127.0.0.1:8000/docs
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Rutas por defecto ──────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
MODEL_PATH  = MODELS_DIR / "best_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURES_PATH = MODELS_DIR / "features.json"

SPLIT_DATE  = "2017-01-01"

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera 24 features temporales, cíclicas, lags y rolling stats."""
    df = df.copy().set_index("Datetime")

    df["hour"]        = df.index.hour
    df["dayofweek"]   = df.index.dayofweek
    df["quarter"]     = df.index.quarter
    df["month"]       = df.index.month
    df["year"]        = df.index.year
    df["dayofyear"]   = df.index.dayofyear
    df["dayofmonth"]  = df.index.day
    df["weekofyear"]  = df.index.isocalendar().week.astype(int)
    df["is_weekend"]  = (df["dayofweek"] >= 5).astype(int)

    # Encoding cíclico
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]     = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Lag features
    for lag in [1, 2, 3, 24, 48, 168]:
        df[f"lag_{lag}h"] = df["PJME_MW"].shift(lag)

    # Rolling statistics
    df["roll_mean_24h"] = df["PJME_MW"].shift(1).rolling(window=24).mean()
    df["roll_std_24h"]  = df["PJME_MW"].shift(1).rolling(window=24).std()
    df["roll_mean_7d"]  = df["PJME_MW"].shift(1).rolling(window=168).mean()

    return df.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def train(data_path: str) -> None:
    """Entrena el mejor modelo (LightGBM) y guarda artefactos en models/."""
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    print("=" * 60)
    print("⚡  Entrenando modelo LightGBM (configuración óptima)")
    print("=" * 60)

    # 1. Carga y preprocesado
    print(f"\n📂 Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = ["Datetime", "PJME_MW"]
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    print(f"   Registros totales: {len(df):,}")

    # 2. Feature engineering
    print("\n🔧 Generando features...")
    df_feat = create_features(df)
    print(f"   Shape tras feature engineering: {df_feat.shape}")

    TARGET   = "PJME_MW"
    FEATURES = [c for c in df_feat.columns if c != TARGET]

    X = df_feat[FEATURES]
    y = df_feat[TARGET]

    # 3. Split temporal (sin shuffle para respetar la causalidad)
    X_train = X[X.index < SPLIT_DATE]
    X_test  = X[X.index >= SPLIT_DATE]
    y_train = y[y.index < SPLIT_DATE]
    y_test  = y[y.index >= SPLIT_DATE]

    print(f"   Train: {X_train.shape[0]:,} filas  |  Test: {X_test.shape[0]:,} filas")
    print(f"   Periodo test: {X_test.index.min()} → {X_test.index.max()}")

    # 4. Mejor configuración LightGBM (según experiments del notebook)
    #    LGB_ne1000_nl127 obtuvo consistentemente el menor RMSE/MAE
    best_params = {
        "n_estimators":      1000,
        "learning_rate":     0.02,
        "num_leaves":        127,
        "min_child_samples": 5,
        "reg_alpha":         0.1,
        "reg_lambda":        0.5,
        "random_state":      42,
        "verbose":           -1,
        "n_jobs":            -1,
    }

    print(f"\n🚀 Entrenando LightGBM con parámetros:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
    )

    # 5. Evaluación
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

    metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2),
               "r2": round(r2, 6), "mape": round(mape, 4),
               "best_iteration": int(model.best_iteration_),
               "n_features": len(FEATURES),
               "train_rows": int(len(X_train)),
               "test_rows":  int(len(X_test)),
               "trained_at": datetime.now().isoformat()}

    print("\n📊 Métricas en test set (2017-2018):")
    print(f"   MAE  : {mae:,.1f} MW")
    print(f"   RMSE : {rmse:,.1f} MW")
    print(f"   R²   : {r2:.4f}")
    print(f"   MAPE : {mape:.2f}%")

    # 6. Guardar artefactos
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURES, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Artefactos guardados en {MODELS_DIR}/")
    print(f"   • best_model.pkl  ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"   • features.json   ({len(FEATURES)} features)")
    print(f"   • metrics.json")
    print("\nAhora puedes levantar la API con:  python energy_api.py serve")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI — SERVICIO DE INFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

def create_app():
    """Crea y configura la aplicación FastAPI."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    # ── Cargar modelo y features al arrancar ──────────────────────────────────
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo no encontrado en {MODEL_PATH}.\n"
            "Entrena primero con:  python energy_api.py train --data <ruta_csv>"
        )

    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH) as f:
        FEATURES = json.load(f)

    saved_metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            saved_metrics = json.load(f)

    # ── Schemas ───────────────────────────────────────────────────────────────
    class PredictionRequest(BaseModel):
        datetime_str:  str   = Field(..., example="2018-06-15 14:00:00",
                                     description="Fecha y hora ISO (YYYY-MM-DD HH:MM:SS)")
        lag_1h:        float = Field(..., example=35200.0, description="Consumo 1 hora antes (MW)")
        lag_2h:        float = Field(..., example=34900.0, description="Consumo 2 horas antes (MW)")
        lag_3h:        float = Field(..., example=34600.0, description="Consumo 3 horas antes (MW)")
        lag_24h:       float = Field(..., example=34100.0, description="Consumo 24 h antes (MW)")
        lag_48h:       float = Field(..., example=33800.0, description="Consumo 48 h antes (MW)")
        lag_168h:      float = Field(..., example=33200.0, description="Consumo 168 h antes / 1 semana (MW)")
        roll_mean_24h: float = Field(..., example=34300.0, description="Media móvil 24 h (MW)")
        roll_std_24h:  float = Field(..., example=750.0,   description="Desviación estándar 24 h (MW)")
        roll_mean_7d:  float = Field(..., example=33950.0, description="Media móvil 7 días (MW)")

    class PredictionResponse(BaseModel):
        datetime_str:  str
        predicted_mw:  float
        model:         str = "LightGBM (best)"

    class BatchRequest(BaseModel):
        inputs: list[PredictionRequest]

    class BatchResponse(BaseModel):
        predictions: list[dict]
        count:       int

    class HealthResponse(BaseModel):
        status:      str
        model_loaded: bool
        metrics:     dict

    # ── Helper ────────────────────────────────────────────────────────────────
    def build_feature_vector(req: PredictionRequest) -> pd.DataFrame:
        dt = pd.Timestamp(req.datetime_str)
        row = {
            "hour":          dt.hour,
            "dayofweek":     dt.dayofweek,
            "quarter":       dt.quarter,
            "month":         dt.month,
            "year":          dt.year,
            "dayofyear":     dt.dayofyear,
            "dayofmonth":    dt.day,
            "weekofyear":    dt.isocalendar()[1],
            "is_weekend":    int(dt.dayofweek >= 5),
            "hour_sin":      np.sin(2 * np.pi * dt.hour / 24),
            "hour_cos":      np.cos(2 * np.pi * dt.hour / 24),
            "month_sin":     np.sin(2 * np.pi * dt.month / 12),
            "month_cos":     np.cos(2 * np.pi * dt.month / 12),
            "dow_sin":       np.sin(2 * np.pi * dt.dayofweek / 7),
            "dow_cos":       np.cos(2 * np.pi * dt.dayofweek / 7),
            "lag_1h":        req.lag_1h,
            "lag_2h":        req.lag_2h,
            "lag_3h":        req.lag_3h,
            "lag_24h":       req.lag_24h,
            "lag_48h":       req.lag_48h,
            "lag_168h":      req.lag_168h,
            "roll_mean_24h": req.roll_mean_24h,
            "roll_std_24h":  req.roll_std_24h,
            "roll_mean_7d":  req.roll_mean_7d,
        }
        return pd.DataFrame([row])[FEATURES]

    # ── App ───────────────────────────────────────────────────────────────────
    app = FastAPI(
        title="⚡ Energy Consumption Predictor",
        description=(
            "Predicción de consumo eléctrico horario (MW) sobre el dataset PJME.\n\n"
            "**Modelo:** LightGBM (LGB_ne1000_nl127) — mejor configuración según "
            "experimentos MLflow.\n\n"
            "**Features requeridas:** fecha/hora + 9 valores históricos de consumo."
        ),
        version="1.0.0",
    )

    # ── Endpoints ─────────────────────────────────────────────────────────────
    @app.get("/", response_model=HealthResponse, summary="Health check + métricas del modelo")
    def root():
        return HealthResponse(
            status="ok",
            model_loaded=True,
            metrics=saved_metrics,
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Predicción individual",
        description="Devuelve la predicción de consumo en MW para una hora concreta.",
    )
    def predict(req: PredictionRequest):
        try:
            X    = build_feature_vector(req)
            pred = float(model.predict(X)[0])
            return PredictionResponse(
                datetime_str=req.datetime_str,
                predicted_mw=round(pred, 2),
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post(
        "/predict/batch",
        response_model=BatchResponse,
        summary="Predicción en batch",
        description="Acepta una lista de hasta 1 000 instancias y devuelve predicciones en bloque.",
    )
    def predict_batch(body: BatchRequest):
        if len(body.inputs) > 1000:
            raise HTTPException(status_code=400, detail="Máximo 1 000 instancias por batch.")
        results = []
        for req in body.inputs:
            X    = build_feature_vector(req)
            pred = float(model.predict(X)[0])
            results.append({"datetime_str": req.datetime_str, "predicted_mw": round(pred, 2)})
        return BatchResponse(predictions=results, count=len(results))

    return app


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    app = create_app()
    print(f"\n🌐 API disponible en http://{host}:{port}")
    print(f"   Docs interactivos: http://127.0.0.1:{port}/docs\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="⚡ Energy Consumption Predictor — train o serve"
    )
    sub = parser.add_subparsers(dest="command")

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Entrenar y guardar el mejor modelo")
    p_train.add_argument(
        "--data", required=True,
        help="Ruta al CSV PJME_hourly.csv (columnas: Datetime, PJME_MW)"
    )

    # ── serve ──────────────────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Levantar la API FastAPI")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "train":
        train(args.data)
    elif args.command == "serve":
        serve(args.host, args.port)
    else:
        parser.print_help()
