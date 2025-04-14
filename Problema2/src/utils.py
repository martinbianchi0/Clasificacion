from .models import RandomForest, MulticlassLogisticRegression
from .metrics import f_score
import numpy as np

def train_best_random_forest(X_train, y_train, X_val, y_val):
    """
    Entrena un Random Forest con los mejores hiperparámetros según F-score (rápido).

    Parámetros:
    - X_train (array o DataFrame): datos de entrenamiento.
    - y_train (array o Series): etiquetas de entrenamiento.
    - X_val (array o DataFrame): datos de validación.
    - y_val (array o Series): etiquetas de validación.

    Retorna:
    - model (RandomForest): modelo entrenado con mejores hiperparámetros.
    """
    best_fscore = 0
    best_params = {}

    # Reducción fuerte del grid
    n_trees_list = [5, 10, 20]
    max_depth_list = [5, 10, 15]
    min_samples_split_list = [2]

    # Usamos un subconjunto de datos para evaluar hiperparámetros
    idx = np.random.choice(len(X_train), size=min(300, len(X_train)), replace=False)
    X_sample = X_train.iloc[idx]
    y_sample = y_train.iloc[idx]

    for n_trees in n_trees_list:
        for max_depth in max_depth_list:
            for min_samples_split in min_samples_split_list:
                model = RandomForest(
                    n_trees=n_trees,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split
                )
                model.fit(X_sample, y_sample)
                y_pred = model.predict(X_val)
                fs = f_score(y_val, y_pred)

                if fs > best_fscore:
                    best_fscore = fs
                    best_params = {
                        'n_trees': n_trees,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split
                    }

    # Entrenar modelo final con todos los datos
    model = RandomForest(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    fs = f_score(y_val, y_pred)
    print(f"Mejor F-score: {fs} con n_trees={best_params['n_trees']}, max_depth={best_params['max_depth']}, min_samples_split={best_params['min_samples_split']}")
    return model

def train_best_multiclass_logistic_model(X_train, y_train, X_val, y_val):
    """Entrena un modelo de regresión logística multiclase (softmax) 
    con los mejores hiperparámetros según el F-score en validación.

    Retorna:
    - model (MulticlassLogisticRegression): modelo entrenado con mejores hiperparámetros.
    """

    best_fscore = 0
    best_params = {'lr': None, 'lambda': None}
    lambdas = [0, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
    learning_rates = [0.1, 0.01, 0.001]

    for lr in learning_rates:
        for l in lambdas:
            model = MulticlassLogisticRegression(X_train, y_train, lr=lr, L2=l)
            y_pred = model.predict(X_val)
            fs = f_score(y_val, y_pred)

            if fs > best_fscore:
                best_fscore = fs
                best_params['lr'] = lr
                best_params['lambda'] = l

    final_model = MulticlassLogisticRegression(X_train, y_train,
                                               lr=best_params['lr'],
                                               L2=best_params['lambda'])
    print(f"Mejor F-score: {best_fscore} con lr={best_params['lr']}, lambda={best_params['lambda']}")
    return final_model
