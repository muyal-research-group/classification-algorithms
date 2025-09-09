import pytest
from classification_algorithms.perceptron import MultiLayer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from axo.contextmanager import AxoContextManager
from typing import Tuple
import numpy.typing as npt

@pytest.fixture()
def example_dataset():
    X, y = make_classification(n_samples=200, n_features=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train,X_test,y_train,y_test

def test_multilayer(example_dataset:Tuple[npt.NDArray,npt.NDArray,npt.NDArray,npt.NDArray]):

    X_train,X_test,y_train,y_test = example_dataset
    
    with AxoContextManager.local() as lr:
        ml : MultiLayer = MultiLayer(
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test
        )
        x = ml.train(axo_endpoint_id = "axo-endpoint-0")
        print("Aqui imprime train",x)
        assert x.is_ok

        y = ml.predict(axo_endpoint_id = "axo-endpoint-0")
        assert y.is_ok
        ml.y_pred = y.unwrap()
        print("Predictions:", ml.y_pred)
        

        metrics_result = ml.get_metrics()
        assert "accuracy" in metrics_result
        assert "classification_report" in metrics_result
        assert 0.0 <= metrics_result["accuracy"] <= 1.0
        print(f"Metrics: {metrics_result['classification_report']}")
        print(f"Precision between 0 and 1:  {metrics_result['accuracy']}")
        
