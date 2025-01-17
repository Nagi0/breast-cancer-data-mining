import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import umap
import lime
import dtreeviz
import plotly.express as px
import matplotlib.pyplot as plt


class Visualizer:

    def visualize_feature_map(self, p_dataset: pd.DataFrame):
        reducer = umap.UMAP(metric="manhattan", random_state=42)
        embeddings = reducer.fit_transform(
            X=p_dataset.drop(columns=["diagnosis"]), y=p_dataset["diagnosis"].to_numpy()
        )

        plot_data = pd.DataFrame(embeddings, columns=["Component1", "Component2"])
        plot_data["Label"] = p_dataset["diagnosis"].replace(1, "Malignant").replace(0, "Benign").to_numpy()

        fig = px.scatter(
            plot_data,
            x="Component1",
            y="Component2",
            color="Label",
            title="Mapa de Atributos do Dataset",
        )

        fig.update_traces(marker=dict(size=6, opacity=1.0))
        fig.update_layout(scene=dict(xaxis_title="embedding_1", yaxis_title="embedding_2"))
        fig.show()

    def confusion_matrix(
        self,
        p_y_test: np.ndarray,
        p_y_hat: np.ndarray,
        p_classifier: SVC | RandomForestClassifier | DecisionTreeClassifier,
    ):
        cm = confusion_matrix(p_y_test, p_y_hat, labels=p_classifier.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=p_classifier.classes_)
        disp.plot()
        plt.show()

    def features_importances(self, p_importances_df: pd.DataFrame):
        imp = p_importances_df.values[0]
        imp, names = zip(*sorted(zip(imp, p_importances_df.columns)))

        plt.figure()
        plt.title("K-Fold Permutation Feature Importance")
        plt.barh(range(len(names)), imp, align="center")
        plt.yticks(range(len(names)), names)
        plt.show()

    def visualize_tree(self, p_classifier, p_X: pd.DataFrame, p_y: pd.Series):
        viz_model = dtreeviz.model(
            p_classifier,
            X_train=p_X.values,
            y_train=p_y.values,
            feature_names=p_X.columns,
            target_name="diagnosis",
            class_names=p_y.unique(),
        )

        v = viz_model.view(fontname="monospace")
        v.save("breastCancerDataMining/Models/tree.svg")

    def visualize_rf_predict(
        self,
        p_explainer: lime.lime_tabular.LimeTabularExplainer,
        p_sample: pd.DataFrame,
        p_predict_fn_rf,
        y_hat: int,
        idx: int,
    ):
        exp = p_explainer.explain_instance(p_sample.values[0], p_predict_fn_rf, num_features=7)
        exp.save_to_file(f"breastCancerDataMining/Models/limeResults/lime_{y_hat}_{idx}.html")
