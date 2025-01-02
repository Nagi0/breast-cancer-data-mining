from dataclasses import dataclass
import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate, train_test_split
from breastCancerDataMining.DataLoading.DataLoader import DataLoader
from breastCancerDataMining.DataLoading.visualizer import Visualizer


@dataclass
class Models:
    dataset: pd.DataFrame
    metrics: dict

    def svm_classfier(self) -> SVC:
        classifier = SVC(kernel="linear", C=3.0, random_state=42)

        return classifier

    def rf_classfier(self) -> RandomForestClassifier:
        classifier = RandomForestClassifier(random_state=42)

        return classifier

    def decision_tree_classifier(self) -> DecisionTreeClassifier:
        classifier = DecisionTreeClassifier(random_state=42)

        return classifier

    def get_model(self, p_model: str):
        if p_model.strip().lower() == "svm":
            classifier = self.svm_classfier()
        elif p_model.strip().lower() == "random forest":
            classifier = self.rf_classfier()
        elif p_model.strip().lower() == "decision tree":
            classifier = self.decision_tree_classifier()

        return classifier

    def select_features(self, p_selected_features: list):
        features = self.dataset.iloc[:, :-1]
        labels = self.dataset.iloc[:, -1]
        features = features[p_selected_features]

        self.dataset = pd.concat([features, labels], axis=1)

    def train_model(self, p_model: str):
        features = self.dataset.iloc[:, :-1]
        labels = self.dataset.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

        classifier = self.get_model(p_model)
        classifier.fit(X_train, y_train)

        y_hat = classifier.predict(X_test)
        Visualizer().confusion_matrix(y_test, y_hat, classifier)

        return classifier

    def get_features_importances(self, p_model: str, p_num_iter: int):
        features = self.dataset.iloc[:, :-1]
        labels = self.dataset.iloc[:, -1]
        classifier = self.get_model(p_model)

        results_df_list = []
        for _ in tqdm(range(p_num_iter)):
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
            classifier.fit(X_train, y_train)

            results = permutation_importance(classifier, X_test, y_test, n_repeats=10).importances_mean
            results_df = pd.DataFrame([results], columns=features.columns)
            results_df["iteration"] = ["iteration"]
            results_df_list.append(results_df)

        features_importance_df = pd.concat(results_df_list).groupby(by="iteration").mean()
        Visualizer().features_importances(features_importance_df)

        imp = features_importance_df.values[0]
        imp, names = zip(*sorted(zip(imp, features_importance_df.columns), reverse=True))

        features_importance_df = pd.DataFrame([imp], columns=names)

        return features_importance_df

    def cross_validation(self, p_model: str, p_cv: int = 5):
        classifier = self.get_model(p_model)

        features = self.dataset.iloc[:, :-1]
        labels = self.dataset.iloc[:, -1]

        scores = cross_validate(classifier, features.values, labels.values, cv=p_cv, scoring=self.metrics, verbose=1)

        return scores


if __name__ == "__main__":
    load_dotenv("./breastCancerDataMining/config/.env")

    data_loader = DataLoader(int(os.environ["dataset_id"]))
    dataset = data_loader.load_dataset()
    dataset = data_loader.standard_normalize_data(dataset)

    metrics_dict = {"Accuracy": "accuracy", "F1-Score": "f1", "Jaccard": "jaccard"}
    modeling = Models(dataset, metrics_dict)
    features_importance_df = modeling.get_features_importances("SVM", 100)

    modeling.select_features(features_importance_df.columns[:7])
    modeling.dataset.to_csv("filtered_dataset.csv", index=False)
    scores = modeling.cross_validation("Decision Tree", p_cv=10)
    print(scores)

    trained_model = modeling.train_model("Decision Tree")
    features = modeling.dataset.iloc[:, :-1]
    labels = modeling.dataset.iloc[:, -1]
    # Visualizer().visualize_tree(trained_model, features, labels)
