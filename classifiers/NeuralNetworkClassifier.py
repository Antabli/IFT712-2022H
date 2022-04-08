
#####
# Réalisé par: Ala Antabli (20012727)
####


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from IPython.display import display

class NN_Classifier(object):
    """
    Classe pour implémenter un modèle de réseau de neurones sur le module Multi Layers Perceptron (MLP) de sklearn.

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

        self.estimator = MLPClassifier(max_iter=800)
        self.scorers = scorers
        self.hyper_search = None

    def default_training(self, verbose=False):
        """
        Entraînons le modèle avec les paramètres sklearn par défaut.

        Retourne un tuple pour:
        - Precision de l'entrainement
        - Précision de la validation
        """
        self.estimator.fit(self.x_train, self.y_train)

        predict = self.estimator.predict(self.x_train)
        accuracy_train = accuracy_score(self.y_train, predict)

        predict_valid = self.estimator.predict(self.x_val)
        accuracy_valid = accuracy_score(self.y_val, predict_valid)

        if verbose:
            print('Precision de l`entrainement {:.3%}'.format(accuracy_train))
            print('Précision de la validation: {:.3%}'.format(accuracy_valid))

        return accuracy_train, accuracy_valid

    def hyperparameter_train(self, grid_param, random_search=True, verbose=False):
        """
        Entraînez le modèle avec un 'Grid Search' et une validation croisée.

        Entrée:
        - grid_param (dict): dictionnaire de valeurs à tester dans la grille de recherche.
        S'il n'est pas fourni, on utilisera les valeurs par défaut de l'estimateur.
        - random_search (bool), default=True -- Si True, on utilise la recherche aléatoire,
                si False recherche toutes les combinaisons de paramètres (prend plus de temps).

        Retourne un tuple pour:
        - L'exactitude d'entrainement
        - L'exactitude de la validation
        Et:
        - Entraine self.estimator avec le meilleur scoreur
        """

        # Initialisation de la Grid search avec kfold
        searching_params = {
            "scoring": self.scorers,
            "refit": "Accuracy",
            "cv": KFold(n_splits=5, shuffle=True),
            "return_train_score": True,
            "verbose": int(verbose),
            "n_jobs": 4}

        if random_search:
            if verbose:
                print("Utilisation de la recherche aléatoire :")
            self.hyper_search = RandomizedSearchCV(self.estimator, grid_param).set_params(**searching_params)
        else:
            if verbose:
                print("Utilisation de la recherche complète :")
            self.hyper_search = GridSearchCV(self.estimator, grid_param).set_params(**searching_params)

        # Recherche de hyper-paramèters
        self.hyper_search.fit(self.x_train, self.y_train)

        # Enregistrons le meilleur estimateur
        self.estimator = self.hyper_search.best_estimator_

        # Prédictions sur les données d'entraînement et de validation
        predict_train = self.hyper_search.predict(self.x_train)
        predict_valid = self.hyper_search.predict(self.x_val)

        # Précision de l'entrainement et de la validation
        accuracy_train = accuracy_score(self.y_train, predict_train)
        accuracy_valid = accuracy_score(self.y_val, predict_valid)

        if verbose:
            print()
            print('Meilleure précision de validation croisée : {}'.format(self.hyper_search.best_score_))
            print('\nMeilleur estimateur:\n{}'.format(self.hyper_search.best_estimator_))
            print()
            print('\nPrécision de l`entrainement: {:.3%}'.format(accuracy_train))
            print('Précision de la validation: {:.3%}'.format(accuracy_valid))

        return accuracy_train, accuracy_valid

    def predict(self, Data):
        """
        Utilisons le modèle formé pour prédire la classe de l'échantillon.
        Data : Une liste contenant un ou plusieurs échantillons.

        Renvoie une étiquette de classe codée pour chaque échantillon.
        """
        return self.estimator.predict(Data)
