from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import matplotlib.pyplot as plt


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy.typing as npt
from typing import Dict, Any
from axo import Axo, axo_method

class LinearRegression(Axo):
    def __init__(
        self, 
        X_train: npt.NDArray, 
        X_test: npt.NDArray, 
        y_train: npt.NDArray, 
        y_test: npt.NDArray, 
        **kwargs
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.modelo = LR(**kwargs)

    @axo_method
    def LinearRegression_train(self, **kwargs) -> Dict[str, Any]:
        fitted = self.modelo.fit(self.X_train, self.y_train)
        return fitted.get_params()

    @axo_method
    def LinearRegression_predict(self, **kwargs) -> npt.NDArray:
        return self.modelo.predict(self.X_test)

    def get_metrics(self, **kwargs) -> Dict[str, Any]:
        y_pred = self.LinearRegression_predict()
        if hasattr(y_pred, "unwrap"): 
            y_pred = y_pred.unwrap()
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {
            "MSE": mse,
            "R2": r2
        }

    def graph_regression(self)-> None:
        y_pred = self.LinearRegression_predict()
        plt.scatter(self.y_test, y_pred, alpha=0.6)
        plt.xlabel("Valores reales")
        plt.ylabel("Predicciones")
        plt.title("Regresión Lineal - Comparación Real vs Predicción")
        plt.show()
        
"""if __name__ == "__main__":
    X, y = make_regression(n_samples=200, n_features=5, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model  = LinearRegression(X_train, X_test, y_train, y_test)
    model .train()
    results = model.metrics()
    print("MSE:", results["MSE"])
    print("R²:", results["R2"])
    model .graph_regression()"""