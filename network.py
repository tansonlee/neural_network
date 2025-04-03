import numpy as np


class NeuralNetworkConfig:
    def __init__(self):
        self.input_nodes = 0
        self.hidden_layers = []
        self.activation = None
        self.cost = None


def activation_sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))


def cost_binary_cross_entropy(y_hat, y):
    losses = - ((y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat))
    m = y_hat.shape[1]
    summed_losses = (1 / m) * np.sum(losses, axis=1)
    return np.sum(summed_losses)


class HiddenLayer:
    def __init__(self, layer_size, prev_layer_size):
        self.layer_size = layer_size
        self.prev_layer_size = prev_layer_size

        self.weights = np.random.randn(layer_size, prev_layer_size)
        self.biases = np.random.randn(layer_size, 1)

    def output(self, prev_layer_values):
        return activation_sigmoid(self.weights @ prev_layer_values + self.biases)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_layer_definitions):
        self.hidden_layers = {}
        for i in range(len(hidden_layer_definitions)):
            neurons = hidden_layer_definitions[i]
            prev_layer_neurons = input_nodes if i == 0 else hidden_layer_definitions[i-1]
            l = HiddenLayer(neurons, prev_layer_neurons)
            self.hidden_layers[i + 1] = l

        self.batch_size = 0
        self.layer_outputs = {}

    def forward(self, input_layer_data):
        data = input_layer_data
        self.layer_outputs[0] = input_layer_data
        self.batch_size = data.shape[1]
        for layer_num in range(1, len(self.hidden_layers) + 1):
            layer = self.hidden_layers[layer_num]
            data = layer.output(data)
            self.layer_outputs[layer_num] = data

        return data

    def _backwards_propagation_output(self, y_hat, Y, m, A2, W3):
        A3 = y_hat

        # step 1. calculate dC/dZ3 using shorthand we derived earlier
        dC_dZ3 = (1/m) * (A3 - Y)

        # step 2. calculate dC/dW3 = dC/dZ3 * dZ3/dW3
        #   we matrix multiply dC/dZ3 with (dZ3/dW3)^T
        dZ3_dW3 = A2

        dC_dW3 = dC_dZ3 @ dZ3_dW3.T

        # step 3. calculate dC/db3 = np.sum(dC/dZ3, axis=1, keepdims=True)
        dC_db3 = np.sum(dC_dZ3, axis=1, keepdims=True)

        # step 4. calculate propagator dC/dA2 = dC/dZ3 * dZ3/dA2
        dZ3_dA2 = W3
        dC_dA2 = dZ3_dA2.T @ dC_dZ3

        return dC_dW3, dC_db3, dC_dA2

    def _backward_propagation_hidden(self, propagator_dC_dA2, A1, A2, W2):
        # step 1. calculate dC/dZ2 = dC/dA2 * dA2/dZ2

        # use sigmoid derivation to arrive at this answer:
        #   sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
        #     and if a = sigmoid(z), then sigmoid'(z) = a * (1 - a)
        dA2_dZ2 = A2 * (1 - A2)
        dC_dZ2 = propagator_dC_dA2 * dA2_dZ2

        # step 2. calculate dC/dW2 = dC/dZ2 * dZ2/dW2
        dZ2_dW2 = A1

        dC_dW2 = dC_dZ2 @ dZ2_dW2.T

        # step 3. calculate dC/db2 = np.sum(dC/dZ2, axis=1, keepdims=True)
        dC_db2 = np.sum(dC_dZ2, axis=1, keepdims=True)

        # step 4. calculate propagator dC/dA1 = dC/dZ2 * dZ2/dA1
        dZ2_dA1 = W2
        dC_dA1 = dZ2_dA1.T @ dC_dZ2

        return dC_dW2, dC_db2, dC_dA1

    def backward(self, expected, step_size):
        curr_layer = len(self.hidden_layers)
        weight_gradient, bias_gradient, propagator = self._backwards_propagation_output(
            self.layer_outputs[curr_layer],
            expected,
            self.batch_size,
            self.layer_outputs[curr_layer - 1],
            self.hidden_layers[curr_layer].weights)

        # Update weights and biases.
        self.hidden_layers[curr_layer].weights = self.hidden_layers[curr_layer].weights - \
            (step_size * weight_gradient)
        self.hidden_layers[curr_layer].biases = self.hidden_layers[curr_layer].biases - \
            (step_size * bias_gradient)

        # Do the rest of the layers
        for curr_layer in reversed(range(1, len(self.hidden_layers))):
            weight_gradient, bias_gradient, propagator = self._backward_propagation_hidden(
                propagator,
                self.layer_outputs[curr_layer - 1],
                self.layer_outputs[curr_layer],
                self.hidden_layers[curr_layer].weights)

            self.hidden_layers[curr_layer].weights = self.hidden_layers[curr_layer].weights - \
                (step_size * weight_gradient)
            self.hidden_layers[curr_layer].biases = self.hidden_layers[curr_layer].biases - \
                (step_size * bias_gradient)
