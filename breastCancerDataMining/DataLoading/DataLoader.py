from dataclasses import dataclass
import os
import pandas as pd
from dotenv import load_dotenv
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from breastCancerDataMining.DataLoading.visualizer import Visualizer


@dataclass
class DataLoader:
    dataset_id: int

    def load_features(self, p_dataset) -> pd.DataFrame:
        X = p_dataset.data.features

        return X

    def load_labels(self, p_dataset) -> pd.DataFrame:
        y = p_dataset.data.targets

        return y

    def load_dataset(self):
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=self.dataset_id)
        X = self.load_features(breast_cancer_wisconsin_diagnostic)
        y = self.load_labels(breast_cancer_wisconsin_diagnostic)
        print(y.value_counts())
        label_encoder = LabelEncoder()
        y = pd.DataFrame(label_encoder.fit_transform(y), columns=["diagnosis"])

        dataset = pd.concat([X, y], axis=1)

        return dataset

    def print_metadata(self):
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=self.dataset_id)

        print(breast_cancer_wisconsin_diagnostic.metadata)
        print(breast_cancer_wisconsin_diagnostic.variables)

    def min_max_normalize_data(self, p_dataset: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler()
        columns_name = p_dataset.columns
        normalized_dataset = pd.DataFrame(
            scaler.fit_transform(p_dataset.drop(columns=[columns_name[-1]])), columns=columns_name[:-1]
        )
        normalized_dataset["diagnosis"] = p_dataset[columns_name[-1]]

        return normalized_dataset

    def standard_normalize_data(self, p_dataset: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        columns_name = p_dataset.columns
        normalized_dataset = pd.DataFrame(
            scaler.fit_transform(p_dataset.drop(columns=[columns_name[-1]])), columns=columns_name[:-1]
        )
        normalized_dataset["diagnosis"] = p_dataset[columns_name[-1]]

        return normalized_dataset


if __name__ == "__main__":
    load_dotenv("./breastCancerDataMining/config/.env")

    data_loader = DataLoader(int(os.environ["dataset_id"]))
    dataset = data_loader.load_dataset()
    dataset = data_loader.standard_normalize_data(
        dataset
    )  # Pela visualuzação no UMAP, há uma dispersão menor no mapa de atributos ao normalizer pelo StandardScaler

    print(dataset)

    Visualizer().visualize_feature_map(dataset)  # Visualização pelo UMAP
