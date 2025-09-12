from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Dict, Any
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.typing as npt
#from sklearn.datasets import load_iris
#from sklearn.model_selection import train_test_split
from axo import Axo, axo_method

class RandomForest(Axo):
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
        self.model = RF(**kwargs)

    @axo_method
    def RandomForest_train(self, **kwargs) -> Dict[str, Any]:
        fitted = self.model.fit(self.X_train, self.y_train)
        return fitted.get_params()

    @axo_method
    def RandomForest_predict(self, **kwargs) -> npt.NDArray:
        return self.model.predict(self.X_test)

    def get_metrics(self, **kwargs) -> Dict[str, Any]:
        y_pred = self.RandomForest_predict()
        if hasattr(y_pred, "unwrap"):
            y_pred = y_pred.unwrap()
        acc = accuracy_score(self.y_test, y_pred)
        cr = classification_report(self.y_test, y_pred)
        return {
            "accuracy": acc,
            "classification_report": cr
        }

    def confussion_matrix(self, **kwargs) -> None:
        
        y_pred = self.RandomForest_predict()
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicción")
        plt.ylabel("Valor real")
        plt.title("Matriz de Confusión - Random Forest")
        plt.show()

    def feature_importances(self, top_n: int = 10) -> Dict[str, float]:

        importances = self.model.feature_importances_
        indices = importances.argsort()[::-1][:top_n]
        return {str(i): importances[i] for i in indices}
    

"""if __name__ == "__main__":
    # Cargar dataset Iris
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )

    # Crear y entrenar el modelo Random Forest
    model = RandomForest(X_train, X_test, y_train, y_test, n_estimators=100, random_state=42)
    model.train()

    # Evaluar el modelo
    results = model.metrics()
    print("Precisión:", results["accuracy"])
    print("Reporte de clasificación:", results["classification_report"])

    # Graficar matriz de confusión
    model.confusion_matrix()

    # Mostrar importancias de características
    print("Importancia de características:", model.feature_importances(top_n=3))"""