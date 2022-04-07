
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

class LogisticRegressionClassifier(object):
    def __init__(self, x_training, y_training, x_valid, y_valid, class_names, scorers):
        self.x_train = x_training
        self.y_train = y_training

        self.x_val = x_valid
        self.y_val = y_valid
        self.class_names = class_names

        self.num_features = x_training.shape[1]
        self.num_classes = class_names.shape

        self.estimator = MultiOutputClassifier(LogisticRegression(), n_jobs=4)
        self.scorers = scorers

    def train_without_grid(self):
        LogisticRegression = self.estimator
        LogisticRegression.fit(self.x_train, self.y_train)
        
        predict = LogisticRegression.predict(self.x_train)
        accuracy_train = accuracy_score(self.y_train, predict)

        predict_valid = LogisticRegression.predict(self.x_val)
        accuracy_valid = accuracy_score(self.y_val, predict_valid)

        return accuracy_train, accuracy_valid

    def train(self, grid_search_params={}, random_search=True):
        """
        Entraînez le modèle avec une implémentation naïve de perte d'entropie croisée (avec boucle)

        Inputs:
        - grid_search_params (dict): dictionnaire de valeurs à tester dans la grille de recherche

        Renvoie un tuple pour :
        - perte d'entrainement
        - perte de validation
        - exactitude d'entrainement
        - exactitude de validation
        """
        # Initialisation de la Grid search avec kfold
        searching_params = {
            "scoring": self.scorers,
            "refit": "Accuracy",
            "cv": KFold(n_splits=5, shuffle=True),
            "return_train_score": True,
            "n_jobs": 4,
            "verbose": 1}

        if random_search:
            print("Utilisation de la recherche aléatoire :")
            search_g = RandomizedSearchCV(self.estimator, grid_search_params).set_params(**searching_params)
        else:
            print("Utilisation de la recherche complète :")
            search_g = GridSearchCV(self.estimator, grid_search_params).set_params(**searching_params)

        # Modèle d'entrainement
        search_g.fit(self.x_train, self.y_train)

        # Enregistrons le meilleur estimateur et imprimons-le avec la meilleure précision obtenue grâce à la validation croisée
        self.estimator = search_g.best_estimator_
        self.best_accuracy = search_g.best_score_
        self.hyper_search = search_g
        
        # Prédictions sur les données d'entraînement et de validation
        predict_train = search_g.predict(self.x_train)
        predict_valid = search_g.predict(self.x_val)
        
        # Précision de l'entrainement et de la validation
        accuracy_train = accuracy_score(self.y_train, predict_train)
        accuracy_valid = accuracy_score(self.y_val, predict_valid)

        return accuracy_train, accuracy_valid, self.estimator, self.best_accuracy

    def predict(self, X):
        """
        Utilisons le modèle formé pour prédire la classe de l'échantillon.
        X : Une liste contenant un ou plusieurs échantillons.

        Renvoie une étiquette de classe codée pour chaque échantillon.
        """
        classLabel = self.estimator.predict(X)
        return classLabel
