import pytest
from classification_algorithms.perceptron import Perceptron
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


def test_perceptron(example_dataset:Tuple[npt.NDArray,npt.NDArray,npt.NDArray,npt.NDArray]):

    X_train,X_test,y_train,y_test = example_dataset
    
    with AxoContextManager.local() as lr:
        p:Perceptron = Perceptron(
            X_train         = X_train,
            X_test          = X_test,
            y_train         = y_train,
            y_test          = y_test,
        )
        x = p.train(axo_endpoint_id = "axo-endpoint-0")
        print(x)
        assert x.is_ok
        y = p.predict(axo_endpoint_id = "axo-endpoint-0")
        print(y)
        assert y.is_ok


        