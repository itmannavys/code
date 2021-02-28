import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris['data'][:, (2, 3)]  # petal length and petal width
y = (iris['target']).astype(np.float64)
print((iris['target'] == 2).astype(np.float64))


def plot_raw_iris():
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'b^', label='setosa')
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'r*', label='versicolor')
    plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], 'gs', label='virginica')
    plt.grid()
    plt.legend(loc='best')
    # plt.savefig('iris_demo')  # , facecolor='w', edgecolor='w', orientation='portrait')
    plt.show()






# svm_clf = Pipeline([
#         ('scaler', StandardScaler()),
#         ('linear_svc', LinearSVC(C=1, loss='hinge', random_state=42))
#     ])
# svm_clf.fit(X, y)
#
# print(svm_clf.predict([[4.9, 1.6]]))
# joblib.dump(svm_clf, 'iris.pkl')
