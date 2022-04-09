
#####
# Réalisé par: Ala Antabli (20012727)
####


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

class LogisticRegression_Classifier(object):
    """
    Classe pour implémenter un modèle d'arbres de décision basé sur l'arbre du module sklearn.

    Paramétres:
    - x_training (array) -- Tableau de valeurs de 'Features' pour l'entrainement.
    - y_training (array) -- Tableau de vrais 'Labels' pour entrainer le modèle.
    - x_valid (array) -- Tableau de valeurs de 'Features' pour valider le modèle.
    - y_valid (array) -- Tableau de vrais de 'Labels' pour valider le modèle.
    - c_names (array) -- Tableau de noms à lier aux 'Labels'
    """
    
    def __init__(self, x_training, y_training, x_valid, y_valid, c_names, scorers):
        self.x_train = x_training
        self.y_train = y_training

        self.x_val = x_valid
        self.y_val = y_valid
        self.class_names = c_names

        self.num_features = x_training.shape[1]
        self.num_classes = c_names.shape

        self.estimator = MultiOutputClassifier(LogisticRegression(), n_jobs=4)
        self.scorers = scorers

    def train_without_grid(self):
        """
        Entrainner le modèle sans 'Grid Search'

        Retourne un tuple pour:
        - Precision de l'entrainement
        - Précision de la validation
        """
        LogisticRegression = self.estimator
        LogisticRegression.fit(self.x_train, self.y_train)
        
        predict = LogisticRegression.predict(self.x_train)
        accuracy_train = accuracy_score(self.y_train, predict)

        predict_valid = LogisticRegression.predict(self.x_val)
        accuracy_valid = accuracy_score(self.y_val, predict_valid)

        return accuracy_train, accuracy_valid

    def train(self, grid_param={}, rand_search=True):
        """
        Entraînez le modèle avec un 'Grid Search' et une validation croisée.

        Entrée:
        - grid_param (dict): dictionnaire de valeurs à tester dans la grille de recherche.
        S'il n'est pas fourni, on utilisera les valeurs par défaut de l'estimateur.
        - rand_search (bool), default=True -- Si True, on utilise la recherche aléatoire,
                si False recherche toutes les combinaisons de paramètres (prend plus de temps).

        Retourne un tuple pour:
        - L'exactitude d'entrainement
        - L'exactitude de la validation
        - Le meilleur estimateur
        - Le meilleur score
        """
        # Initialisation de la Grid search avec kfold
        search_parameter = {
            "scoring": self.scorers,
            "refit": "Accuracy",
            "cv": KFold(n_splits=5, shuffle=True),
            "return_train_score": True,
            "n_jobs": 4,
            "verbose": 1}

        if rand_search:
            print("Utilisation de la recherche aléatoire :")
            search_grid = RandomizedSearchCV(self.estimator, grid_param).set_params(**search_parameter)
        else:
            print("Utilisation de la recherche complète :")
            search_grid = GridSearchCV(self.estimator, grid_param).set_params(**search_parameter)

        # Entrainement
        search_grid.fit(self.x_train, self.y_train)

        # Enregistrons le meilleur estimateur et imprimons-le avec la meilleure précision obtenue grâce à la validation croisée
        self.estimator = search_grid.best_estimator_
        self.best_accuracy = search_grid.best_score_
        self.hyper_search = search_grid
        
        # Prédictions sur les données d'entraînement et de validation
        predict_train = search_grid.predict(self.x_train)
        predict_valid = search_grid.predict(self.x_val)
        
        # Précision de l'entrainement et de la validation
        accuracy_train = accuracy_score(self.y_train, predict_train)
        accuracy_valid = accuracy_score(self.y_val, predict_valid)

        return accuracy_train, accuracy_valid, self.estimator, self.best_accuracy

    def predict(self, Data):
        """
        Utilisons le modèle formé pour prédire la classe de l'échantillon.
        Data : Une liste contenant un ou plusieurs échantillons.

        Renvoie une étiquette de classe codée pour chaque échantillon.
        """
        return self.estimator.predict(Data)
