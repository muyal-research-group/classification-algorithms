from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import Dict, Any
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.typing as npt
from axo import Axo, axo_method


class KNNClassifier(Axo):
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
        self.model = KNN(**kwargs)

    @axo_method
    def Knn_train(self, **kwargs) -> Dict[str, Any]:
        fitted = self.model.fit(self.X_train, self.y_train)
        return fitted.get_params()

    @axo_method
    def Knn_predict(self, **kwargs) -> npt.NDArray:
        return self.model.predict(self.X_test)

    def get_metrics(self, **kwargs) -> Dict[str, Any]:
        y_pred = self.predict()
        if hasattr(y_pred, "unwrap"):
            y_pred = y_pred.unwrap()
        acc = accuracy_score(self.y_test, y_pred)
        cr = classification_report(self.y_test, y_pred)
        return {
            "accuracy": acc,
            "classification_report": cr
        }

    def confussion_matrix(self, **kwargs) -> None:
        y_pred = self.predict()
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicciones")
        plt.ylabel("Valores reales")
        plt.title("Matriz de Confusión - KNN")
        plt.show()

"""if __name__ == "__main__":
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    model = KNNClassifier(X_train, X_test, y_train, y_test, n_neighbors=5)
    model.train()

    results = model.metrics()
    print("Accuracy:", results["accuracy"])
    print("Reporte de clasificación:", results["classification_report"])

    model.confusion_matrix()"""