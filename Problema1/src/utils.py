import numpy as np
from .metrics import f_score
from Problema1.src.models import LogisticRegression 


def train_best_logistic_model(X_train, y_train, X_val, y_val, weigths=None):
    """Entrena un modelo de regresión logística con los mejores hiperparámetros 
    según el F-score en el set de validación.

    Parámetros:
    - X_train, y_train: datos y etiquetas de entrenamiento.
    - X_val, y_val: datos y etiquetas de validación.
    - weights (array, opcional): pesos por muestra.

    Retorna:
    - model (LogisticRegression): modelo entrenado con los mejores hiperparámetros."""
    best_fscore = 0
    best_params = {'lr': None, 'lambda': None, 'threshold': None}
    lambdas = [0, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
    learning_rates = [0.1, 0.01, 0.001]
    thresholds = np.linspace(0, 1, 100)

    for lr in learning_rates:
        for l in lambdas:
            model = LogisticRegression(X_train, y_train, lr=lr, L2=l, weights=weigths)
            y_scores = model.predict_proba(X_val)

            for t in thresholds:
                y_pred = (y_scores >= t).astype(int)
                fs = f_score(y_val, y_pred)

                if fs > best_fscore:
                    best_fscore = fs
                    best_params['lr'] = lr
                    best_params['lambda'] = l
                    best_params['threshold'] = t
    model = LogisticRegression(X_train, y_train, lr=best_params['lr'], threshold=best_params['threshold'], L2=best_params['lambda'], weights=weigths)
    return model
