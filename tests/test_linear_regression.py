import pytest
from classification_algorithms.linear_regression import LinearRegression 
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
        lr : LinearRegression = LinearRegression(
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test
        )
        x = lr.train(axo_endpoint_id = "axo-endpoint-0")
        assert x.is_ok

        y = lr.predict(axo_endpoint_id = "axo-endpoint-0")
        assert y.is_ok
        lr.y_pred = y.unwrap()
        print("Predictions:", lr.y_pred)
        

        metrics_result = lr.get_metrics()
        assert "MSE" in metrics_result
        assert "R2" in metrics_result
        print(f"Mean Square Error (MSE): {metrics_result["MSE"]}")
        print(f"Coefficient of determination (R2): {metrics_result["R2"]}")
        
