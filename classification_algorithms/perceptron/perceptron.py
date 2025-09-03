from sklearn.linear_model import Perceptron as P
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Dict,Any
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.typing as npt
from axo import Axo,axo_method
# from numpy as np

class Perceptron(Axo):
    def __init__(self, 
        X_train:npt.NDArray,
        X_test:npt.NDArray, 
        y_train:npt.NDArray, 
        y_test:npt.NDArray, 
        **kwargs
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.modelo = P(**kwargs)

    @axo_method
    def train(self,**kwargs)->Dict[str,Any]:
        x = self.modelo.fit(self.X_train, self.y_train)
        return x.get_params()

    @axo_method
    def predict(self,**kwargs):
        return self.modelo.predict(self.X_test)

    def get_metrics(self,**kwargs):
        y_pred = self.predict()
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("\nReporte de Clasificación:\n", classification_report(self.y_test, y_pred))

    def confussion_matrix(self,**kwargs):
        y_pred = self.predict()
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.title("Matriz de Confusión - Perceptrón Simple")
        plt.show()

# if __name__ == "__main__":
#     X, y = make_classification(n_samples=200, n_features=8, random_state=42)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     modelo = ModeloPerceptronSimple(X_train, X_test, y_train, y_test, max_iter=1000)
#     modelo.entrenar()
#     modelo.metricas()
#     modelo.matriz_confusion()