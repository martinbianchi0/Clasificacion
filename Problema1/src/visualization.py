import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from .metrics import pr_auc, roc_auc

def explore_data(df):
    """Imprime un resumen del dataset combinado: muestra aleatoria, rango de valores,
    columnas con nulos y cantidad de duplicados.

    Parámetros:
    - df (DataFrame): conjunto de datos a explorar."""
    print("Fragmento aleatorio de muestras")
    display(df.sample(7))
    print("\nRango de valores de cada columna")
    display(df.describe().loc[['min', 'max']])
    print("\nCategorías con valores faltantes\n", df.isna().sum()[df.isna().sum() > 0].to_string())
    print("\nFilas duplicadas:", df.duplicated().sum())

def plot_correlation_matrix(df):
    """Muestra un mapa de calor con la correlación entre variables numéricas.

    Parámetros:
    - df (DataFrame): dataset a analizar."""
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, bin_cols=None):
    """Genera boxplots para columnas numéricas no binarias.

    Parámetros:
    - df (DataFrame): dataset a analizar.
    - bin_cols (list of str): columnas a excluir por ser binarias."""
    if bin_cols is None:
        bin_cols = []

    num_cols = [col for col in df.select_dtypes(include='number').columns if col not in bin_cols]
    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(y=df[col].reset_index(drop=True), ax=axes[i])
        axes[i].set_title(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_pr(y_true, y_pred):
    """Grafica la curva Precisión-Recall del modelo y el área bajo la curva.

    Parámetros:
    - y_true (Series): etiquetas verdaderas.
    - y_pred (Series): probabilidades predichas."""
    recalls, precisions, auc_pr = pr_auc(y_true, y_pred)
    plt.plot(recalls, precisions, label=f'Curva PR (AUC={auc_pr:.4f})')
    plt.fill_between(recalls, precisions, alpha=0.3, label='Área bajo la curva')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva de Precisión-Recall')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roc(y_true, y_pred):
    """Grafica la curva ROC del modelo y el área bajo la curva.

    Parámetros:
    - y_true (Series): etiquetas verdaderas.
    - y_pred (Series): probabilidades predichas."""
    fpr, tpr, auc_roc = roc_auc(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'Curva ROC (AUC={auc_roc:.4f})')
    plt.fill_between(fpr, tpr, alpha=0.3, label='Área bajo la curva')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_all_pr(resultados, y_val):
    """Grafica curvas Precisión-Recall de múltiples modelos sobre el mismo conjunto de validación.

    Parámetros:
    - resultados (list of dict): cada dict debe incluir 'Modelo' y 'y_scores'.
    - y_val (Series): etiquetas verdaderas del conjunto de validación."""
    plt.figure(figsize=(8, 6))
    for result in resultados:
        nombre = result["Modelo"]
        y_scores = result["y_scores"]
        recalls, precisions, auc_pr = pr_auc(y_val, y_scores)
        plt.plot(recalls, precisions, label=f'{nombre} (AUC-PR={auc_pr:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curvas PR de los modelos')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()


def plot_all_roc(resultados, y_val):
    """Grafica curvas ROC de múltiples modelos sobre el mismo conjunto de validación.

    Parámetros:
    - resultados (list of dict): cada dict debe incluir 'Modelo' y 'y_scores'.
    - y_val (Series): etiquetas verdaderas del conjunto de validación."""
    plt.figure(figsize=(8, 6))
    for result in resultados:
        nombre = result["Modelo"]
        y_scores = result["y_scores"]
        fpr, tpr, auc_roc = roc_auc(y_val, y_scores)
        plt.plot(fpr, tpr, label=f'{nombre} (AUC-ROC={auc_roc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Línea aleatoria')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC de los modelos')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()