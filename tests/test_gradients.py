# test backward passes (including activation layers)

import numpy as np
from src.activations import Softmax
from src.losses import CrossEntropyLoss


def test_softmax_gradient(x, grad_output):
    #x = np.array([[1.0, 0.2, -3.0], [2.0, -1.0, 0.1], [5.0, 3.0, -3.0]])
    #grad_output = np.array([[-0.5, 1.2, 5.0], [1.9, -2.2, 0.7], [-2.0, 0.75, 3.0]])
    softmax = Softmax()
    softmax.forward(x)
    numerical_grad = np.zeros_like(x)
    analytic_grad = softmax.backward(grad_output)

    def loss_function(y):
        p = softmax.forward(y)
        return np.sum(p * grad_output)


    # fill in numerical gradients
    epsilon = 1e-5
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i,j] += epsilon
            x_minus[i,j] -= epsilon
            loss_plus = loss_function(x_plus)
            loss_minus = loss_function(x_minus)
            numerical_grad[i,j] = (loss_plus - loss_minus) / (2 * epsilon)
    # print(numerical_grad)
    # print(analytic_grad)
    np.testing.assert_allclose(numerical_grad, analytic_grad, rtol=1e-5, atol=1e-7)

def test_ce_with_softmax_gradient(x, y):
    softmax = Softmax()
    ce = CrossEntropyLoss()
    p = softmax.forward(x)
    ce.forward(p,y)
    grad_ce = ce.backward()
    grad_x = softmax.backward(grad_ce)
    row_indices = np.arange(p.shape[0])
    one_matrix = np.zeros_like(p)
    one_matrix[row_indices, y] = 1
    simplified_grad = (p - one_matrix)/x.shape[0]
    np.testing.assert_allclose(grad_x, simplified_grad, rtol=1e-5, atol=1e-7)