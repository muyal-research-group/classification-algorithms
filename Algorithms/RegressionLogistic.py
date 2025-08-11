from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class ModeloRegresionLogistica:
    def __init__(self, X_train, X_test, y_train, y_test, **kwargs):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.modelo = LogisticRegression(**kwargs)

    def entrenar(self):
        self.modelo.fit(self.X_train, self.y_train)

    def predecir(self):
        return self.modelo.predict(self.X_test)

    def metricas(self):
        y_pred = self.predecir()
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("\nReporte de Clasificación:\n", classification_report(self.y_test, y_pred))

    def matriz_confusion(self):
        y_pred = self.predecir()
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.title("Matriz de Confusión - Regresión Logística")
        plt.show()


if __name__ == "__main__":
    X, y = make_classification(n_samples=200, n_features=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = ModeloRegresionLogistica(X_train, X_test, y_train, y_test, max_iter=200)
    modelo.entrenar()
    modelo.metricas()
    modelo.matriz_confusion()