import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
#from sklearn.datasets import make_moons
#from sklearn.model_selection import train_test_split
import numpy.typing as npt
from typing import Dict, Any
from axo import Axo, axo_method

class SVM(Axo):
    def __init__(
        self, 
        X_train: npt.NDArray, 
        X_test: npt.NDArray, 
        y_train: npt.NDArray, 
        y_test: npt.NDArray, 
        **kwargs
    ):
        # Se fuerza kernel='rbf' para transformar los datos
        kwargs.setdefault('kernel', 'rbf')
        kwargs.setdefault('C', 1.0)
        kwargs.setdefault('gamma', 'scale')
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.modelo = SVC(**kwargs)

    @axo_method
    def Svm_train(self, **kwargs) -> Dict[str, Any]:
        self.modelo.fit(self.X_train, self.y_train)

    @axo_method
    def Svm_predict(self, **kwargs) -> npt.NDArray:
        return self.modelo.predict(self.X_test)

    def get_metrics(self, **kwargs) -> Dict[str, Any]:
        y_pred = self.Svm_predict()
        if hasattr(y_pred, "unwrap"):
            y_pred = y_pred.unwrap()
        acc = accuracy_score(self.y_test, y_pred)
        cr = classification_report(self.y_test, y_pred)
        return {
            "accuracy": acc,
            "classification_report": cr
        }

    def confussion_matrix(self, **kwargs) -> None:
        y_pred = self.Svm_predict()
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.title("Matriz de Confusión - SVM")
        plt.show()

    # @axo_method
    def graph_border(self)-> None:
        paso = 0.01
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, paso),
                             np.arange(y_min, y_max, paso))

        puntos = np.c_[xx.ravel(), yy.ravel()]
        Z = self.modelo.predict(puntos).reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=plt.cm.coolwarm, edgecolors='k')
        plt.axis('equal')
        plt.title("Frontera de decisión - SVM (kernel rbf)")
        plt.show()

"""if __name__ == "__main__":
    X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = ModeloSVM(X_train, X_test, y_train, y_test)
    model.train()
    results = model.metrics()
    print("Accuracy:", results["accuracy"])
    print("Reporte de clasificación:", results["classification_report"])
    model.confusion_matrix()
    model.graph_border()
"""