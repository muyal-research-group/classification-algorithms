# Axo - Classification Algorithms

This repository provides a collection of **classification algorithms** implemented in Python with a unified structure.  
All algorithms are integrated with [Axo](https://github.com/muyal-research-group/axo/tree/dev) to provide **standardized training, prediction, metrics, and visualization**.  

The project is designed to:  
- Ensure **consistent method names** across all  decorated algorithms (`train`, `predict`).  
- Use **typing** (`npt.NDArray`, `Dict`, `Any`, etc.) everywhere possible.  
- Be fully testable using **pytest**.  

## 📦 Installation

### Option 1: Using [Poetry](https://python-poetry.org/)
```bash
poetry install

To install from the prebuilt package:
```bash
poetry add axo==0.0.1a12 
```

⚠️ please use the newest version of Axo you can check it  https://github.com/muyal-research-group/axo


---

## 📂 Project Structure  
```
.
├── README.md
├── classification_algorithms
│ ├── decission_tree.py
│ ├── linear_regression.py
│ ├── logistic_regression.py
│ ├── naive_bayes.py
│ ├── perceptron
│ │ ├── init.py
│ │ ├── multicapa.py
│ │ └── perceptron.py
│ └── svm.py
├── poetry.lock
├── pyproject.toml
└── tests
├── init.py
└── test_perceptron.py
```

## ⚡ Example Usage  

The `Perceptron` implementation serves as the **reference example**.  
All other algorithms (`DecisionTree`, `LogisticRegression`, `NaiveBayes`, `SVM`, etc.) will follow the same pattern.  

```python
from classification_algorithms.perceptron.perceptron import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from axo.contextmanager import AxoContextManager

# Generate dataset
X, y = make_classification(n_samples=200, n_features=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

with AxoContextManager.local() as lr:
    # Initialize model 
    p = Perceptron(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # Train
    params = p.train()
    print("Model parameters:", params)

    # Predict
    y_pred = p.predict()
    print("Predictions:", y_pred)
```

## 🧪 Testing

We use pytest for testing all classification algorithms.

Run the tests with:
```bash 
pytest ./tests -s -vv
```

### 📏 Code Coverage
To evaluate how much of the codebase is exercised by the test suite, we use ```coverage```

```sh
coverage run -m pytest tests/
coverage report -m
```

- ```coverage run``` wraps the pytest invocation to trace executed statements.

- ```coverage report -m``` generates a summary including missing lines and module

This allows us to:

- Identify untested logic or edge cases.

- Focus testing efforts on high-risk areas.

- Prevent regressions by maintaining coverage thresholds over time.

You can also generate an HTML report for visual inspection:

```sh
coverage html
```
Then open ```htmlcov/index.html``` in your browser to explore annotated source files with coverage information.

## License

This project is licensed under the MIT License.
