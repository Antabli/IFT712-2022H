
#####
# Réalisé par: Ala Antabli (20012727)
####

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt


class Afficher:
    def __init__(self, model) -> None:
        self.model = model
        self.results = model.hyper_search.cv_results_

    def hyper_results(self):
        df = pd.DataFrame.from_dict(self.model.hyper_search.cv_results_, orient='index').rename_axis(
            'Splits results', axis=0).rename_axis('Split number', axis=1)
        return df

    def best_N_estimators(self, TopN=3):
        my_scorer = "Accuracy"
        for i in range(1, TopN + 1):
            all_candidates = np.flatnonzero(self.results["rank_test_{}".format(my_scorer)] == i)
            for candidate in all_candidates:
                print("Modèle avec rang: {0}".format(i))
                print(
                    "Validation moyenne: {0}: {1:.3f} (std: {2:.3f})".format(
                        my_scorer,
                        self.results["mean_test_{}".format(my_scorer)][candidate],
                        self.results["std_test_{}".format(my_scorer)][candidate],
                    )
                )
                print("Paraméters: {0}".format(self.results["params"][candidate]))
                print("")

    def reporting_class(self, y_true, y_predict, mean_only=True):
        my_report = classification_report(y_true, y_predict, target_names=self.model.class_names, zero_division=0, output_dict=True)
        my_report = pd.DataFrame.from_dict(my_report, orient='index')
        if mean_only:
            my_report = my_report.iloc[-4]
        return my_report

    def make_plot(self, abs_params):
        """
        Inspiré de https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html 
        """
        plt.figure(figsize=(13, 8))

        plt.title("Evaluation GridSearchCV pour {}".format(self.model.__class__.__name__), fontsize=16)

        plt.xlabel(abs_params)
        plt.ylabel("Score")
        size = max(self.results["param_{}".format(abs_params)].data)
        ax = plt.gca()
        ax.set_xlim(0, size)
        ax.set_ylim(0, 1.01)

        # Obtenons le tableau numpy régulier du MaskedArray
        axis_X = np.array(self.results["param_{}".format(abs_params)].data, dtype=float)

        for scorers, colors in zip(sorted(self.model.scorers), ["g", "k", "r", "b"]):
            for sample, style in (("train", "--"), ("test", "-")):
                sample_score_mean = self.results["mean_%s_%s" % (sample, scorers)]
                sample_score_std = self.results["std_%s_%s" % (sample, scorers)]
                ax.fill_between(
                    axis_X,
                    sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1 if sample == "test" else 0,
                    color=colors,
                )
                ax.plot(
                    axis_X,
                    sample_score_mean,
                    style,
                    color=colors,
                    alpha=1 if sample == "test" else 0.7,
                    label="%s (%s)" % (scorers, sample),
                )

            best_index = np.nonzero(self.results["rank_test_%s" % scorers] == 1)[0][0]
            best_score = self.results["mean_test_%s" % scorers][best_index]

            # Tracez une ligne verticale pointillée au meilleur score pour ce 'scorer' marqué par x
            ax.plot(
                [
                    axis_X[best_index],
                ]
                * 2,
                [0, best_score],
                linestyle="-.",
                color=colors,
                marker="x",
                markeredgewidth=3,
                ms=8,
            )

            # Annotez le meilleur score pour ce 'scorer'
            ax.annotate("%0.2f" % best_score, (axis_X[best_index], best_score + 0.005))

        plt.legend(loc="best")
        plt.grid(False)
        plt.savefig("./graphs/{}.png".format(self.model.__class__.__name__), transparent=False)
        plt.show()
