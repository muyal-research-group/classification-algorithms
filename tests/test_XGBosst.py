import pytest
from classification_algorithms.XGBosst import XGBoost
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from axo.contextmanager import AxoContextManager
from typing import Tuple
import numpy.typing as npt

@pytest.fixture()
def dataset_example():
    X, y = make_classification(n_samples=200, n_features=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def test_xgbosst(dataset_example : Tuple[npt.NDArray,npt.NDArray,npt.NDArray,npt.NDArray]):
    X_train, X_test, y_train, y_test = dataset_example
    with AxoContextManager.local() as Ir:
        xg = XGBoost(
            X_train = X_train, 
            X_test=X_test, 
            y_train =y_train, 
            y_test=y_test
        )
        x = xg.train(axo_edpoint_id = "axo-edpoint-0")
        assert x.is_ok

        y = xg.predict(axp_edpoint_id = "axo-edpoint-0")
        assert y.is_ok
        xg.y_pred = y.unwrap()
        print("Predictions:", xg.y_pred)

        metrics_result = xg.get_metrics()
        assert "accuracy" in metrics_result
        assert "classification_report" in metrics_result
        assert 0.0 <= metrics_result["accuracy"] <= 1.0
        print(f"Metrics: {metrics_result["classification_report"]}")
        print(f"Precision between 0 and 1: {metrics_result["accuracy"]}")