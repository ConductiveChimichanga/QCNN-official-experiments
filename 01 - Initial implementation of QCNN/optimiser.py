import numpy as np

#use adam optimiser instead of nestrov, since it requires less tuning

class Adam:
    #matches hyperparameter defaults used by pennylane implemenaatation
    def __init__(self, n_params, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = np.zeros(n_params)   #first moment - the mean of the gradients
        self.v     = np.zeros(n_params)   #second moment - the variance
        self.step_count = 0

    def step(self, params, grad):
        self.step_count += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat  = self.m / (1 - self.beta1 ** self.step_count)  #bias correction
        v_hat  = self.v / (1 - self.beta2 ** self.step_count)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def numerical_gradient(loss_fn, params, eps=0.01):
    #central-difference gradient — two loss evals per parameter
    grad = np.zeros_like(params)
    for i in range(len(params)):
        p_plus, p_minus = params.copy(), params.copy()
        p_plus[i]  += eps
        p_minus[i] -= eps
        grad[i] = (loss_fn(p_plus) - loss_fn(p_minus)) / (2 * eps)
    return grad
