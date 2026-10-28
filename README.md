from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "MLE (no reg)": LogisticRegression(C=1e6, max_iter=1000),
    "MAP L2 Ridge": LogisticRegression(C=1.0, penalty='l2', max_iter=1000),
    "MAP L1 Lasso": LogisticRegression(C=1.0, penalty='l1', solver='liblinear', max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred))
    zeros = (model.coef_ == 0).sum()
    print(f"Zero coefficients: {zeros}")

mle = LogisticRegression(C=1e6, max_iter=1000).fit(X_train, y_train)
l2 = LogisticRegression(C=1.0, penalty='l2', max_iter=1000).fit(X_train, y_train)
l1 = LogisticRegression(C=1.0, penalty='l1', solver='liblinear', max_iter=1000).fit(X_train, y_train)

idx = np.arange(X_train.shape[1])
plt.figure(figsize=(12, 4))
plt.bar(idx - 0.25, mle.coef_[0], 0.25, label='MLE')
plt.bar(idx, l2.coef_[0], 0.25, label='L2')
plt.bar(idx + 0.25, l1.coef_[0], 0.25, label='L1')
plt.xlabel("Feature Index")
plt.ylabel("Coefficient")
plt.title("MLE vs L1 vs L2 Coefficients")
plt.legend()
plt.show()
