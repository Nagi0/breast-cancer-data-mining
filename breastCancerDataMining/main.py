import os
import lime
import lime.lime_tabular
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from breastCancerDataMining.Models.models import Models
from breastCancerDataMining.DataLoading.data_loader import DataLoader
from breastCancerDataMining.DataLoading.visualizer import Visualizer


if __name__ == "__main__":
    load_dotenv("./breastCancerDataMining/config/.env")

    data_loader = DataLoader(int(os.environ["dataset_id"]))
    dataset = data_loader.load_dataset()
    dataset = data_loader.standard_normalize_data(dataset)

    metrics_dict = {"Accuracy": "accuracy", "F1-Score": "f1", "Jaccard": "jaccard"}
    modeling = Models(dataset, metrics_dict)
    features_importance_df = modeling.get_features_importances("SVM", 100)

    modeling.select_features(features_importance_df.columns[:7])
    modeling.dataset.to_csv("breastCancerDataMining/Models/filtered_dataset.csv", index=False)
    scores = modeling.cross_validation("Decision Tree", p_cv=10)
    print(scores)

    trained_model = modeling.train_model("Decision Tree")
    features = modeling.dataset.iloc[:, :-1]
    labels = modeling.dataset.iloc[:, -1]
    # Visualizer().visualize_tree(trained_model, features, labels)

    rf_model = modeling.train_model("Random Forest")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

    predict_fn_rf = lambda x: rf_model.predict_proba(x).astype(float)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, feature_names=X_train.columns, class_names=["B", "M"]
    )

    for idx, sample in enumerate(X_test.values):
        sample = pd.DataFrame(sample.reshape((1, -1)), columns=X_test.columns)
        y_hat = rf_model.predict(sample)[0]
        if y_hat == y_test.values[idx]:
            Visualizer().visualize_rf_predict(explainer, sample, predict_fn_rf, y_hat, idx)
