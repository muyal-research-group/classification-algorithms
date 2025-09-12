from classification_algorithms.perceptron import MultiLayer, Perceptron
from classification_algorithms.decission_tree import DecisionTree
from classification_algorithms.knn import KNNClassifier
from classification_algorithms.linear_regression import LinearRegression
from classification_algorithms.logistic_regression import LogisticRegression
from classification_algorithms.naive_bayes import NaiveBayes
from classification_algorithms.random_forest import RandomForest
from classification_algorithms.svm import SVM
from classification_algorithms.xgboost import XGBoost
from axo.contextmanager import AxoContextManager
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
params = X_train, X_test, y_train, y_test


def benchmark_perceptron(runs=31):
    with AxoContextManager.local() as Ir:
        p: Perceptron = Perceptron(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        for i in range(runs):
                x = p.Perceptron_train(axo_endpoint_id="axo-endpoint-0")
                assert x.is_ok
        for i in range(runs):
                y = p.Perceptron_predict(axo_endpoint_id="axo-endpoint-0")
                assert y.is_ok
        

def benchmark_multilayer(runs=31):
    with AxoContextManager.local() as Ir:
        ml: MultiLayer = MultiLayer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        for i in range(runs):
            x = ml.Multilayer_train(axo_endpoint_id="axo-endpoint-0")
            assert x.is_ok
        for i in range(runs):
            y = ml.Multilayer_predict(axo_endpoint_id="axo-endpoint-0")
            assert y.is_ok

def benchmark_decissiontree(runs=31):
    with AxoContextManager.local() as Ir:
        dt: DecisionTree = DecisionTree(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        for i in range(runs):
            x = dt.DecissionTree_train(axo_endpoint_id="axo-endpoint-0")
            assert x.is_ok
        for i in range(runs):
            y = dt.DecissionTree_predict(axo_endpoint_id="axo-endpoint-0")
            assert y.is_ok

def benchmark_knn(runs=31):
    with AxoContextManager.local() as Ir:
        k: KNNClassifier = KNNClassifier(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        for i in range(runs):
            x = k.Knn_train(axo_endpoint_id="axo-endpoint-0")
            assert x.is_ok
        for i in range(runs):
            y = k.Knn_predict(axo_endpoint_id="axo-endpoint-0")
            assert y.is_ok

def benchmark_linear(runs=31):
    with AxoContextManager.local() as Ir:
        lr: LinearRegression = LinearRegression(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        for i in range(runs):
            x = lr.LinearRegression_train(axo_endpoint_id="axo-endpoint-0")
            assert x.is_ok
        for i in range(runs):
            y = lr.LinearRegression_predict(axo_endpoint_id="axo-endpoint-0")
            assert y.is_ok

def benchmark_logistic(runs=31):
    with AxoContextManager.local() as Ir:
        lr: LogisticRegression = LogisticRegression(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        for i in range(runs):
            x = lr.LogisticRegression_train(axo_endpoint_id="axo-endpoint-0")
            assert x.is_ok
        for i in range(runs):
            y = lr.LogisticRegression_predict(axo_endpoint_id="axo-endpoint-0")
            assert y.is_ok

def benchmark_naive(runs=31):
    with AxoContextManager.local() as Ir:
        nb: NaiveBayes = NaiveBayes(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        for i in range(runs):
            x = nb.NaiveBayes_train(axo_endpoint_id="axo-endpoint-0")
            assert x.is_ok
        for i in range(runs):
            y = nb.NaiveBayes_predict(axo_endpoint_id="axo-endpoint-0")
            assert y.is_ok

def benchmark_rf(runs=31):
    with AxoContextManager.local() as Ir:
        rf: RandomForest = RandomForest(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        for i in range(runs):
            x = rf.RandomForest_train(axo_endpoint_id="axo-endpoint-0")
            assert x.is_ok
        for i in range(runs):
            y = rf.RandomForest_predict(axo_endpoint_id="axo-endpoint-0")
            assert y.is_ok

def benchmark_svm(runs=31):
    with AxoContextManager.local() as Ir:
        s: SVM = SVM(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        for i in range(runs):
            x = s.Svm_train(axo_endpoint_id="axo-endpoint-0")
            assert x.is_ok
        for i in range(runs):
            y = s.Svm_predict(axo_endpoint_id="axo-endpoint-0")
            assert y.is_ok

def benchmark_xgboost(runs=31):
    with AxoContextManager.local() as Ir:
        xgb: XGBoost = XGBoost(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        for i in range(runs):
            x = xgb.XgBosst_train(axo_endpoint_id="axo-endpoint-0")
            assert x.is_ok
        for i in range(runs):
            y = xgb.XgBosst_predict(axo_endpoint_id="axo-endpoint-0")
            assert y.is_ok


if __name__ == "__main__":
    benchmark_perceptron()
    benchmark_multilayer()
    benchmark_decissiontree()
    benchmark_knn()
    benchmark_linear()
    benchmark_logistic()
    benchmark_naive()
    benchmark_rf()
    benchmark_svm()
    benchmark_xgboost()

    print("Benchmarks completados para todos los algoritmos")