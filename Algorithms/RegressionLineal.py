from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

class ModeloRegresionLineal:
    def __init__(self, X_train, X_test, y_train, y_test, **kwargs):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.modelo = LinearRegression(**kwargs)

    def entrenar(self):
        self.modelo.fit(self.X_train, self.y_train)

    def predecir(self):
        return self.modelo.predict(self.X_test)

    def metricas(self):
        y_pred = self.predecir()
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")

    def grafico_regresion(self):
        y_pred = self.predecir()
        plt.scatter(self.y_test, y_pred, alpha=0.6)
        plt.xlabel("Valores reales")
        plt.ylabel("Predicciones")
        plt.title("Regresión Lineal - Comparación Real vs Predicción")
        plt.show()
if __name__ == "__main__":
    X, y = make_regression(n_samples=200, n_features=5, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = ModeloRegresionLineal(X_train, X_test, y_train, y_test)
    modelo.entrenar()
    modelo.metricas()
    modelo.grafico_regresion()