import os
import lime
import lime.lime_tabular
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import f_oneway
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.model_selection import train_test_split
from statsmodels.multivariate.manova import MANOVA
from breastCancerDataMining.Models.models import Models
from breastCancerDataMining.DataLoading.data_loader import DataLoader
from breastCancerDataMining.DataLoading.visualizer import Visualizer


if __name__ == "__main__":
    load_dotenv("./breastCancerDataMining/config/.env")

    data_loader = DataLoader(int(os.environ["dataset_id"]))
    dataset = data_loader.load_dataset()
    dataset = data_loader.standard_normalize_data(dataset)

    metrics_dict = {"Accuracy": "accuracy", "F1-Score": "f1", "Jaccard": "jaccard", "ROC AUC": "roc_auc"}
    modeling = Models(dataset, metrics_dict)
    features_importance_df = modeling.get_features_importances("SVM", 100)

    modeling.select_features(features_importance_df.columns[:7])
    modeling.dataset.to_csv("breastCancerDataMining/Models/filtered_dataset.csv", index=False)
    scores = modeling.cross_validation("Decision Tree", p_cv=10)
    print(f"Decision Tree Cross Validation Scores:\n{scores}\n")

    trained_model = modeling.train_model("Decision Tree")
    features = modeling.dataset.iloc[:, :-1]
    labels = modeling.dataset.iloc[:, -1]
    # Visualizer().visualize_tree(trained_model, features, labels)

    rf_model = modeling.train_model("Random Forest")
    scores = modeling.cross_validation("Random Forest", p_cv=10)
    print(f"Random Forest Cross Validation Scores:\n{scores}\n")
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

    subgroups = {
        "area3 = < -0.324393": {"data": dataset.loc[dataset["area3"] < -0.324393], "attributes": ["area3"]},
        "texture3 = < -0.378974, radius3 = -0.2825 - 0.108886": {
            "data": dataset.loc[
                (dataset["texture3"] < -0.378974) & (dataset["radius3"] >= -0.2825) & (dataset["radius3"] <= 0.108886)
            ],
            "attributes": ["texture3", "radius3"],
        },
        "concavity3 = -0.267647 - 0.451502, area2 = < -0.480123": {
            "data": dataset.loc[
                (dataset["concavity3"] >= -0.267647)
                & (dataset["concavity3"] <= 0.451502)
                & (dataset["area2"] < -0.480123)
            ],
            "attributes": ["concavity3", "area2"],
        },
        "texture3 = ≥ 0.589132, concavity3 = < -0.267647": {
            "data": dataset.loc[(dataset["texture3"] >= 0.589132) & (dataset["concavity3"] < -0.267647)],
            "attributes": ["texture3", "concavity3"],
        },
        "area3 = < -0.324393, area2 = -0.199161 - 0.295767": {
            "data": dataset.loc[
                (dataset["area3"] < -0.324393) & (dataset["area2"] >= -0.199161) & (dataset["area2"] <= 0.295767)
            ],
            "attributes": ["area3", "area2"],
        },
        "area3 = < -0.324393, concavity1 = 0.00779267 - 0.38168": {
            "data": dataset.loc[
                (dataset["area3"] < -0.324393)
                & (dataset["concavity1"] >= 0.00779267)
                & (dataset["concavity1"] <= 0.38168)
            ],
            "attributes": ["area3", "concavity1"],
        },
        "area3 = -0.324393 - 0.00697341, area2 = < -0.480123": {
            "data": dataset.loc[
                (dataset["area3"] >= -0.324393) & (dataset["area3"] <= 0.00697341) & (dataset["area2"] < -0.480123)
            ],
            "attributes": ["area3", "area2"],
        },
        "area3 = < -0.324393, concavity1 = ≥ 0.38168": {
            "data": dataset.loc[(dataset["area3"] < -0.324393) & (dataset["concavity1"] >= 0.38168)],
            "attributes": ["area3", "concavity1"],
        },
    }
    for name in subgroups.keys():
        # print(
        #     f"{name}:\n{subgroups[name]['data']}{dataset.iloc[~dataset.index.isin(subgroups[name]['data'].index)]}\n"
        # )
        sb_attributes = subgroups[name]["attributes"]
        subgroup_df = subgroups[name]["data"][sb_attributes]
        subgroup_df = subgroup_df.assign(group="subgroup")
        subgroup_complement_df = dataset.iloc[~dataset.index.isin(subgroups[name]["data"].index)][sb_attributes]
        subgroup_complement_df = subgroup_complement_df.assign(group="complement")

        combined_df = pd.concat([subgroup_df, subgroup_complement_df])

        formula = ""
        for idx, attrib in enumerate(sb_attributes):
            if idx > 0:
                formula = f"{formula} + {attrib}"
            else:
                formula = f"{attrib}"

        formula = f"{formula} ~ group"
        print(formula)

        if len(sb_attributes) > 1:
            manova = MANOVA.from_formula(formula=formula, data=combined_df)
            print(manova.mv_test())
        else:
            stat, p_value = ttest_ind(
                subgroup_df[sb_attributes], subgroup_complement_df[sb_attributes], equal_var=False
            )
            print(f"stats: {stat}\np-value: {p_value}\n")

            stat_mwu, p_value_mwu = mannwhitneyu(subgroup_df[sb_attributes], subgroup_complement_df[sb_attributes])
            print(f"stats_mwu: {stat}\np_value_mwu: {p_value_mwu}\n")

            f_result = f_oneway(subgroup_df[sb_attributes], subgroup_complement_df[sb_attributes])
            print(f_result)
