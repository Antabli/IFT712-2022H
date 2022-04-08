
#####
# Réalisé par: Ala Antabli (20012727)
####


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import pandas as pd
from sklearn.svm import SVC

class SVM_Classifier(object):
    """
    Classe pour implémenter un modèle SVM sur le module de sklearn.

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

        self.estimator = MultiOutputClassifier(SVC(probability=True), n_jobs=4)
        self.scorers = scorers

    def train_without_grid(self):
        """
        Entrainner le modèle sans 'Grid Search'

        Retourne un tuple pour:
        - Precision de l'entrainement
        - Précision de la validation
        """
        svm = self.estimator
        svm.fit(self.x_train, self.y_train)
        predict = svm.predict(self.x_train)
        accuracy_train = accuracy_score(self.y_train, predict)
        
        predict_valid = svm.predict(self.x_val)
        accuracy_valid = accuracy_score(self.y_val, predict_valid)

        return accuracy_train, accuracy_valid

    def train(self, grid_param={}, random_search=True):
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
        - Le meilleur estimateur
        - Le meilleur score
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
            search_g = RandomizedSearchCV(self.estimator, grid_param).set_params(**searching_params)
        else:
            print("Utilisation de la recherche complète :")
            search_g = GridSearchCV(self.estimator, grid_param).set_params(**searching_params)

        # Entrainement
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

    def predict(self, Data):
        """
        Use the trained model to predict the sample's class.
        Data: A list containing one or many samples.

        Returns a encoded class label for each sample.
        """
        class_lab = self.estimator.predict(Data)
        return class_lab