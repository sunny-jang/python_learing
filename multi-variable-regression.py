import numpy as np

from simple_regression import learning_rate
from utils import numerical_derivative

loaded_data = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[:, :-1]
y_data = loaded_data[:, [-1]]

w = np.random.randn(3, 1)
b = np.random.randn(1)


print(x_data)
print(y_data)

def loss_func(x, t):
    y = np.dot(x, w) + b
    return (np.mean(t - y) ** 2) / len(x)


def predict(x):
    return np.dot(x, w) + b


learning_rate = 1e-5

f = lambda x: loss_func(x_data, y_data)

print("initial error value", loss_func(x_data, y_data), "initial w", w, "initial b", b)

for step in range(10000):
    w -= learning_rate * numerical_derivative(f, w)
    b -= learning_rate * numerical_derivative(f, b)

    if step % 400 == 0:
        print("initial error value", loss_func(x_data, y_data), "initial w", w, "initial b", b)


test_data = np.array([100,98,91]).reshape(1, -1)
print(predict(test_data))