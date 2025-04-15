# Trabajo Práctico 2 - Clasificación y Ensemble Learning

**Materia:** I302 - Aprendizaje Automático y Aprendizaje Profundo  
**Semestre:** 1er Semestre 2025  
**Fecha de entrega:** Lunes 14 de abril de 2025, 23:59 hs  
**Formato de entrega:** Archivo comprimido `.zip` en el Campus Virtual  
**Lenguajes/Librerías permitidas:**  
- **NumPy** para cálculos matriciales y funciones numéricas.  
- **Pandas** para manipulación de datos.  
- **Matplotlib/Seaborn** para visualización de datos.  
- **No está permitido el uso de librerías de Machine Learning como scikit-learn.**  

---

## 📌 Descripción

Este trabajo práctico aborda dos problemas de clasificación:

- **Problema 1: Diagnóstico de Cáncer de Mama.**  
  Clasificación binaria sobre datos clínicos de biopsias, incluyendo técnicas para el tratamiento de datos desbalanceados y una implementación propia de regresión logística con regularización.

- **Problema 2: Predicción de Rendimiento de Jugadores de Basketball.**  
  Clasificación multiclase para predecir la categoría de impacto de jugadores en base a estadísticas. Se utilizan tres modelos: Regresión Logística Multiclase, Análisis Discriminante Lineal (LDA) y Random Forest.

## 📂 Estructura del Proyecto

Apellido_Nombre_TP2/
├── Apellido_Nombre_Informe.pdf
├── Apellido_Nombre_Notebook_TP2.ipynb
├── Problema1/
│   ├── data/
│   │   ├── raw/
│   │   │   ├── cell_diagnosis_dev.csv
│   │   │   ├── cell_diagnosis_test.csv
│   │   │   ├── cell_diagnosis_dev_imbalanced.csv
│   │   │   ├── cell_diagnosis_test_imbalanced.csv
│   │   │   └── cell_diagnosis_description.md
│   │   └── processed/
│   │       ├── cell_diagnosis_dev.csv
│   │       └── cell_diagnosis_test.csv
│   └── src/
│       ├── models.py
│       ├── metrics.py
│       ├── preprocessing.py
│       ├── visualization.py
│       └── utils.py
├── Problema2/
│   ├── data/
│   │   ├── raw/
│   │   │   ├── WAR_class_dev.csv
│   │   │   ├── WAR_class_test.csv
│   │   │   └── WAR_class.md
│   │   └── processed/
│   │       ├── WAR_class_dev.csv
│   │       └── WAR_class_test.csv
│   └── src/
│       ├── models.py
│       ├── metrics.py
│       ├── preprocessing.py
│       ├── visualization.py
│       └── utils.py
├── requirements.txt
└── README.md

---

## 📊 Contenido del Trabajo

### 1️⃣ Diagnóstico de Cáncer de Mama  
- Análisis exploratorio completo (outliers, NaNs, categóricas).  
- Imputación de valores con KNN basada en contexto multivariable.  
- Normalización con **RobustScaler** y **MinMaxScaler** según distribución.  
- Implementación de regresión logística binaria con regularización L2.  
- Evaluación con métricas completas: Accuracy, F1, AUC, PR y ROC.  
- Comparación de estrategias de rebalanceo:  
  - Sin rebalanceo  
  - Undersampling  
  - Oversampling  
  - SMOTE  
  - Cost re-weighting  
- Evaluación final sobre test y discusión de resultados.  

### 2️⃣ Predicción de Rendimiento en Basketball  
- Limpieza y análisis de variables (detección de variables derivadas, outliers).  
- Eliminación de features redundantes y normalización con RobustScaler.  
- Entrenamiento de tres modelos:  
  - **Random Forest** (con entropía)  
  - **Logistic Regression Multiclase**  
  - **LDA (Análisis Discriminante Lineal)**  
- Búsqueda de hiperparámetros y evaluación en conjunto de validación.  
- Re-entrenamiento sobre train+validation y evaluación final en test.  
- Análisis comparativo entre modelos. Random Forest fue el más robusto.

---

## 🛠 Instalación y Ejecución

1. Descomprimir el archivo `.zip`:
   ```sh
   unzip Apellido_Nombre_TP2.zip
   cd Apellido_Nombre_TP2
