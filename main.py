from network import NeuralNetwork, cost_binary_cross_entropy
from sklearn.datasets import load_breast_cancer
import numpy as np

nn = NeuralNetwork(30, [128, 64, 1])


def get_xor_data():
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    expected = np.array([[0, 1, 1, 0]])
    return data, expected


def get_breast_cancer_data():
    all_data = load_breast_cancer()

    data = all_data.data.T
    data = data / np.max(data, axis=1, keepdims=True)
    expected = all_data.target.reshape(1, -1)
    return data, expected


data, expected = get_breast_cancer_data()
print(data)
print(expected)

for i in range(50_000):
    result = nn.forward(data)
    cost = cost_binary_cross_entropy(result, expected)
    nn.backward(expected, 0.2)
    if i % 1000 == 0:
        print("cost", i, cost)

print("data", data)
print("result", result)
print("expected", expected)
print("difference", expected - result)
print("cost", cost)
