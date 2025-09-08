import pytest
from classification_algorithms.svm import SVM
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from axo.contextmanager import AxoContextManager
from typing import Tuple
import numpy.typing as npt

@pytest.fixture()
def dataset_example():
    X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def test_svm(dataset_example : Tuple[npt.NDArray,npt.NDArray,npt.NDArray,npt.NDArray]):
    X_train, X_test, y_train, y_test = dataset_example
    with AxoContextManager.local() as Ir:
        svm = SVM(
            X_train     =X_train, 
            X_test  =X_test, 
            y_train =y_train, 
            y_test  =y_test
        )
        x = svm.train(axo_edpoint_id = "axo-edpoint-0")
        assert x.is_ok

        y = svm.predict(axo_edpoint_id = "axo-edpoint-0")
        assert y.is_ok
        svm.y_pred = y.unwrap()
        print("Predictions:", svm.y_pred) 

        metrics_results = svm.get_metrics()
        assert "accuracy" in metrics_results
        assert "classification_report" in metrics_results
        assert 0.0 <= metrics_results["accuracy"] <= 1.0
        print(f"Metrics:{metrics_results["classification_report"]}")
        print(f"Precision between 0 and 1:  {metrics_results["accuracy"]}")