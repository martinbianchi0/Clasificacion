import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def precision(y_true, y_pred):
    clases = np.unique(y_true)
    precisiones = []
    for c in clases:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        p = tp / (tp + fp) if (tp + fp) != 0 else -1
        precisiones.append(p)
    return np.mean([p for p in precisiones if p != -1])

def recall(y_true, y_pred):
    clases = np.unique(y_true)
    recalls = []
    for c in clases:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        r = tp / (tp + fn) if (tp + fn) != 0 else -1
        recalls.append(r)
    return np.mean([r for r in recalls if r != -1])

def f_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p == -1 or r == -1 or (p + r) == 0:
        return -1
    return 2 * (p * r) / (p + r)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def matriz_de_confusion(y_true, y_pred):
    clases = np.unique(np.concatenate([y_true, y_pred]))
    n = len(clases)
    matriz = np.zeros((n, n), dtype=int)
    for i, actual in enumerate(clases):
        for j, predicho in enumerate(clases):
            matriz[i, j] = np.sum((y_true == actual) & (y_pred == predicho))
    return matriz

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# def pr_auc(y_true, y_scores):
#     """
#     Calcula la curva Precisión-Recall y el área bajo la curva (AUC-PR) para problemas multiclase.

#     Parámetros:
#     - y_true (array-like): Etiquetas verdaderas.
#     - y_scores (array-like): Matriz de probabilidades predichas (n_samples x n_classes).

#     Retorna:
#     - dict: Diccionario donde cada clave es una clase, y el valor es una tupla (recalls, precisions, auc_pr).
#     """
#     clases = np.unique(y_true)
#     resultados = {}

#     # Mapear valor de clase a índice en y_scores
#     clase_to_idx = {clase: idx for idx, clase in enumerate(clases)}

#     for c in clases:
#         y_true_bin = (y_true == c).astype(int)
#         idx = clase_to_idx[c]  # índice correspondiente
#         y_scores_clase = y_scores[:, idx]
        
#         thresholds = np.sort(np.unique(y_scores_clase))[::-1]
#         precisions = []
#         recalls = []
        
#         for t in thresholds:
#             y_pred = (y_scores_clase >= t).astype(int)
#             tp = np.sum((y_pred == 1) & (y_true_bin == 1))
#             fp = np.sum((y_pred == 1) & (y_true_bin == 0))
#             fn = np.sum((y_pred == 0) & (y_true_bin == 1))

#             precision = tp / (tp + fp) if (tp + fp) != 0 else -1
#             recall = tp / (tp + fn) if (tp + fn) != 0 else -1

#             precisions.append(precision)
#             recalls.append(recall)
        
#         recalls = [0.0] + recalls
#         precisions = [1.0] + precisions

#         auc_pr = np.trapz(precisions, recalls)
#         resultados[c] = (recalls, precisions, auc_pr)
    
#     return resultados

# def roc_auc(y_true, y_scores):
#     """
#     Calcula la curva ROC y el área bajo la curva (AUC-ROC) para problemas multiclase.

#     Parámetros:
#     - y_true (array-like): Etiquetas verdaderas.
#     - y_scores (array-like): Matriz de probabilidades predichas (n_samples x n_classes).

#     Retorna:
#     - dict: Diccionario donde cada clave es una clase, y el valor es una tupla (fpr, tpr, auc_roc).
#     """
#     clases = np.unique(y_true)
#     resultados = {}

#     clase_to_idx = {clase: idx for idx, clase in enumerate(clases)}

#     for c in clases:
#         y_true_bin = (y_true == c).astype(int)
#         idx = clase_to_idx[c]
#         y_scores_clase = y_scores[:, idx]


#         thresholds = np.linspace(0, 1, 100)
#         tpr = []
#         fpr = []

#         for t in thresholds:
#             y_pred = (y_scores_clase >= t).astype(int)
#             tp = np.sum((y_pred == 1) & (y_true_bin == 1))
#             fp = np.sum((y_pred == 1) & (y_true_bin == 0))
#             tn = np.sum((y_pred == 0) & (y_true_bin == 0))
#             fn = np.sum((y_pred == 0) & (y_true_bin == 1))

#             tpr.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
#             fpr.append(fp / (fp + tn) if (fp + tn) != 0 else 0)

#         auc_roc = np.trapz(tpr, fpr)
#         resultados[c] = (fpr, tpr, auc_roc)
    
#     return resultados

def pr_auc(y_true, y_scores):
    """
    Calcula la curva Precisión-Recall y el área bajo la curva (AUC-PR) combinando
    todos los verdaderos y falsos positivos de todas las clases (micro-promedio).

    Parámetros:
    - y_true: array de etiquetas verdaderas (n_samples,)
    - y_scores: array (n_samples, n_clases) de probabilidades predichas

    Retorna:
    - recalls: lista de recall para cada umbral
    - precisions: lista de precision para cada umbral
    - auc_pr: área bajo la curva precision-recall
    """
    clases = np.unique(y_true)
    n_samples, n_classes = y_scores.shape

    # Convertir y_true a one-hot
    y_true_onehot = np.zeros_like(y_scores)
    for i, c in enumerate(clases):
        y_true_onehot[:, i] = (y_true == c).astype(int)

    # Aplanar ambos arrays para combinar todos los valores
    y_true_flat = y_true_onehot.ravel()
    y_scores_flat = y_scores.ravel()

    # Ordenar por score descendente
    orden = np.argsort(-y_scores_flat)
    y_true_sorted = y_true_flat[orden]
    y_scores_sorted = y_scores_flat[orden]

    tp = 0
    fp = 0
    precisions = []
    recalls = []

    total_positivos = np.sum(y_true_flat)

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / total_positivos

        precisions.append(precision)
        recalls.append(recall)

    # Agregamos punto inicial
    recalls = [0.0] + recalls
    precisions = [1.0] + precisions

    auc_pr = np.trapz(precisions, recalls)
    return recalls, precisions, auc_pr

def roc_auc(y_true, y_scores):
    """
    Calcula la curva ROC y el área bajo la curva (AUC-ROC) combinando
    todos los verdaderos y falsos positivos de todas las clases (micro-promedio).

    Parámetros:
    - y_true: array de etiquetas verdaderas (n_samples,)
    - y_scores: array (n_samples, n_clases) de probabilidades predichas

    Retorna:
    - fpr: lista de tasa de falsos positivos
    - tpr: lista de tasa de verdaderos positivos
    - auc_roc: área bajo la curva ROC
    """
    clases = np.unique(y_true)
    n_samples, n_classes = y_scores.shape

    # Convertir y_true a one-hot
    y_true_onehot = np.zeros_like(y_scores)
    for i, c in enumerate(clases):
        y_true_onehot[:, i] = (y_true == c).astype(int)

    # Aplanar
    y_true_flat = y_true_onehot.ravel()
    y_scores_flat = y_scores.ravel()

    # Ordenar por score descendente
    orden = np.argsort(-y_scores_flat)
    y_true_sorted = y_true_flat[orden]

    tp = 0
    fp = 0
    fn = np.sum(y_true_flat)
    tn = len(y_true_flat) - fn

    tpr = []
    fpr = []

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1

        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr.append(tpr_val)
        fpr.append(fpr_val)

    auc_roc = np.trapz(tpr, fpr)
    return fpr, tpr, auc_roc


# def plot_all_metrics(y_true, y_scores, y_pred):
#     """
#     Grafica métricas de desempeño para un modelo multiclase:
#     - Curvas PR y ROC para cada clase.
#     - Matriz de confusión y métricas resumen.

#     Parámetros:
#     - y_true (array-like): Etiquetas verdaderas.
#     - y_scores (array-like): Matriz de probabilidades predichas (n_samples x n_classes).
#     - y_pred (array-like): Etiquetas predichas.
#     """
#     # Obtener resultados
#     resultados_pr = pr_auc(y_true, y_scores)
#     resultados_roc = roc_auc(y_true, y_scores)
#     cm = matriz_de_confusion(y_true, y_pred)

#     clases = np.unique(y_true)
#     n_clases = len(clases)

#     # Crear figura
#     fig, axes = plt.subplots(2, 2, figsize=(14, 12))

#     # === Curva PR Micro ===
#     recalls, precisions, auc_pr = pr_auc(y_true, y_scores)
#     axes[0, 0].plot(recalls, precisions, label=f'Micro-PR (AUC={auc_pr:.4f})')
#     axes[0, 0].set_xlabel('Recall')
#     axes[0, 0].set_ylabel('Precision')
#     axes[0, 0].set_title('Curva Precisión-Recall (Micro)')
#     axes[0, 0].legend()
#     axes[0, 0].grid(True)

#     # === Curva ROC Micro ===
#     fpr, tpr, auc_roc = roc_auc(y_true, y_scores)
#     axes[0, 1].plot(fpr, tpr, label=f'Micro-ROC (AUC={auc_roc:.4f})')
#     axes[0, 1].plot([0, 1], [0, 1], linestyle='--', color='grey', label='Línea aleatoria')
#     axes[0, 1].set_xlabel('FPR')
#     axes[0, 1].set_ylabel('TPR')
#     axes[0, 1].set_title('Curva ROC (Micro)')
#     axes[0, 1].legend()
#     axes[0, 1].grid(True)


#     # === Matriz de Confusión ===
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases, ax=axes[1, 0])
#     axes[1, 0].set_xlabel('Predicción')
#     axes[1, 0].set_ylabel('Realidad')
#     axes[1, 0].set_title('Matriz de Confusión')

#     # === Métricas Resumen ===
#     axes[1, 1].axis('off')
#     metricas = {
#         "Accuracy": round(accuracy(y_true, y_pred), 3),
#         "F-Score": round(f_score(y_true, y_pred), 3),
#         "Precision": round(precision(y_true, y_pred), 3),
#         "Recall": round(recall(y_true, y_pred), 3),
#     }
#     tabla = [[k, v] for k, v in metricas.items()]
#     tabla_ax = axes[1, 1].table(cellText=tabla, colLabels=["Métrica", "Valor"], cellLoc='center', loc='center')
#     tabla_ax.auto_set_font_size(False)
#     tabla_ax.set_fontsize(12)
#     tabla_ax.scale(1.5, 2)
#     axes[1, 1].set_title('Resumen de Métricas')

#     plt.tight_layout()
#     plt.show()

def compare_models_metrics(y_vals, y_probs, y_preds, nombres=["LDA", "Logistic", "Random Forest"]):
    """
    Compara múltiples modelos graficando:
    - Curva PR
    - Curva ROC
    - Tabla resumen de métricas (Accuracy, F1, Precision, Recall)

    Parámetros:
    - y_vals: lista de arrays con etiquetas verdaderas.
    - y_probs: lista de arrays de probabilidades predichas.
    - y_preds: lista de arrays con etiquetas predichas.
    - nombres: lista de nombres de los modelos.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # === 1. Curva PR ===
    for i in range(len(y_vals)):
        recalls, precisions, auc_pr = pr_auc(y_vals[i], y_probs[i])
        axes[0].plot(recalls, precisions, label=f'{nombres[i]} (AUC={auc_pr:.3f})')
    axes[0].set_title("Curva Precisión-Recall")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend()
    axes[0].grid(True)

    # === 2. Curva ROC ===
    for i in range(len(y_vals)):
        fpr, tpr, auc_roc = roc_auc(y_vals[i], y_probs[i])
        axes[1].plot(fpr, tpr, label=f'{nombres[i]} (AUC={auc_roc:.3f})')
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[1].set_title("Curva ROC")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].legend()
    axes[1].grid(True)

    # === 3. Tabla de métricas ===
    resultados = []
    for i in range(len(y_vals)):
        y_true = y_vals[i]
        y_pred = y_preds[i]
        resultados.append({
            "Modelo": nombres[i],
            "Accuracy": round(accuracy(y_true, y_pred), 3),
            "Precision": round(precision(y_true, y_pred), 3),
            "Recall": round(recall(y_true, y_pred), 3),
            "F1": round(f_score(y_true, y_pred), 3),
        })
    tabla = pd.DataFrame(resultados)

    # Insertar tabla
    axes[2].axis('off')
    table_data = tabla.values.tolist()
    col_labels = tabla.columns.tolist()
    table = axes[2].table(cellText=table_data, colLabels=col_labels, 
                          cellLoc='center', loc='center')

    # Estilo
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        else:
            cell.set_facecolor('#f9f9f9')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.8, 2.5)
    axes[2].set_title('Resumen de Métricas')

    plt.tight_layout()
    plt.show()


def compare_confusion_matrices(y_vals, y_preds, nombres=["LDA", "Logistic", "Random Forest"]):
    """
    Muestra las matrices de confusión de múltiples modelos en una figura 1x3.

    Parámetros:
    - y_vals: lista de arrays con etiquetas verdaderas.
    - y_preds: lista de arrays con etiquetas predichas.
    - nombres: lista de nombres de los modelos.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i in range(len(y_vals)):
        cm = matriz_de_confusion(y_vals[i], y_preds[i])
        clases = np.unique(y_vals[i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases, ax=axes[i])
        axes[i].set_title(f'Matriz de Confusión - {nombres[i]}')
        axes[i].set_xlabel('Predicción')
        axes[i].set_ylabel('Realidad')

    plt.tight_layout()
    plt.show()
