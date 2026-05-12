# ⚡ PJME Energy Consumption Predictor (MLOps Challenge)

Este repositorio contiene un flujo completo de **MLOps** para la predicción del consumo eléctrico horario en la región Este de PJM (EE. UU.), integrando experimentación con **MLflow**, entrenamiento de modelos de alto rendimiento y despliegue mediante una API lista para producción con **FastAPI**.

## 🚀 Descripción del Proyecto
El objetivo es predecir la demanda energética en Megavatios (MW) basándose en el dataset histórico **PJM Hourly Energy Consumption**. El proyecto asegura la trazabilidad de los experimentos y facilita la puesta en producción del modelo ganador.

### Características principales:
* **Dataset:** ~145,000 registros horarios (2002-2018).
* **Ingeniería de variables:** Creación de variables temporales, codificación cíclica (sin/cos), lags históricos y medias móviles.
* **Modelado:** Comparación de algoritmos de regresión (LightGBM, XGBoost, Random Forest).
* **MLflow Tracking:** Registro detallado de parámetros, métricas (MAE, RMSE, R²) y gestión de modelos en el Registry.
* **Producción:** Servicio de inferencia escalable desarrollado con FastAPI y Uvicorn.

---

## 📁 Estructura del Repositorio
* `energy_api.py`: Script principal para el entrenamiento del modelo y el servicio de la API.
* `energy_consumption_mlops.ipynb`: Notebook de Jupyter con el análisis exploratorio y la experimentación inicial.
* `models/`: Carpeta con los artefactos del modelo (generada tras el entrenamiento).
* `requirements.txt`: Lista de dependencias del proyecto.
* `README.md`: Documentación del repositorio.

---

## 🛠️ Instalación y Configuración

1. **Clonar el repositorio:**
   git clone https://github.com/TU_USUARIO/mlops-energy-consumption.git
   cd mlops-energy-consumption

2. **Crear y activar entorno virtual:**
   python3 -m venv venv
   source venv/bin/activate

3. **Instalar dependencias:**
   pip install -r requirements.txt

---

## 📈 Uso del Proyecto

### 1. Entrenamiento del Modelo
Para entrenar el modelo LightGBM configurado con los parámetros óptimos:
python energy_api.py train --data data/PJME_hourly.csv

### 2. Despliegue de la API
Para iniciar el servidor de predicción:
python energy_api.py serve

El servidor estará activo en http://localhost:8000.

### 3. Documentación Interactiva
Accede a la interfaz de **Swagger UI** para probar predicciones individuales o en batch:
👉 http://127.0.0.1:8000/docs

---

## 📊 Gestión con MLflow
Se utilizó la interfaz de MLflow para guiar la selección de hiperparámetros, observando la evolución del RMSE frente a cambios en la complejidad del modelo. El modelo final fue seleccionado y marcado como `Production` por su excelente balance entre precisión y generalización.

Para abrir el panel de experimentos:
mlflow ui --port 5000

---

## ✒️ Autor
* **Luis** - [luislopezperez401](https://github.com/luislopezperez401)
