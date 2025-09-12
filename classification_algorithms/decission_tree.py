from sklearn.tree import DecisionTreeClassifier as DTC, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import Dict, Any
from axo import Axo, axo_method

class DecisionTree(Axo):
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
        self.modelo = DTC(**kwargs)

    @axo_method
    def DecissionTree_train(self, **kwargs)-> Dict[str, Any]:
        fitted = self.modelo.fit(self.X_train, self.y_train)
        return fitted.get_params()

    @axo_method
    def DecissionTree_predict(self, **kwargs)-> npt.NDArray:
        return self.modelo.predict(self.X_test)

    def get_metrics(self) -> Dict[str, Any]:
        y_pred = self.DecissionTree_predict()
        if hasattr(y_pred, "unwrap"):
            y_pred = y_pred.unwrap()
        acc = accuracy_score(self.y_test, y_pred)
        cr = classification_report(self.y_test, y_pred)
        return {
            "accuracy": acc,
            "classification_report": cr
        }

    def confussion_matrix(self, **kwargs)-> None:
        y_pred = self.DecissionTree_predict()
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.title("Matriz de Confusión - Árbol de Decisión")
        plt.show()

    def graph_tree(self) -> None:
        plt.figure(figsize=(12, 8))
        plot_tree(self.modelo, filled=True, feature_names=[f"X{i}" for i in range(self.X_train.shape[1])])
        plt.show()
    

"""if __name__ == "__main__": 
    X, y = make_classification(n_samples=200, n_features=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = DecisionTree(X_train, X_test, y_train, y_test, max_depth=4)
    model.train()
    
    results = model.metrics()
    print("Accuracy:", results["accuracy"])
    print("Reporte de Clasificación:", results["classification_report"])

    model.confusion_matrix()
    model.graph_tree()"""