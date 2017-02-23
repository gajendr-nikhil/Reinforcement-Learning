import numpy as np
from copy import deepcopy

class TrainingData():
    def __init__(self, first=0, last=6, step=1):
        self.basis_vector = {}
        for i in range(first, last + 1, step):
            self.basis_vector[i] = np.eye(last + 1)[i]

    def one_sequence(self, start=3, goal=[0,6]):
        seq, curr = self.basis_vector[start], start
        while True:
            next = (curr + 1) if np.random.rand() > 0.5 else (curr - 1)
            seq = np.vstack((seq, self.basis_vector[next]))
            if next in goal:
                break
            curr = next
        return seq

    def get_training_data(self, num=100, seq=10):
        data = []
        for i in range(num):
            oneseq = []
            for j in range(seq):
                oneseq.append(self.one_sequence())
            data.append(oneseq)
        return data

class Model():
    def __init__(self, alpha=0.01, lamb_da=0, epsilon=0.1):
        self.alpha, self.lamb_da, self.epsilon = alpha, lamb_da, epsilon
        self.calculated_w = np.array([0., 0.5, 0.5, 0.5, 0.5, 0.5, 1.])
        self.ideal_w = np.array([0., 1./6, 2./6, 3./6, 4./6, 5./6, 1.])

    def fit(self, train):
        raise NotImplementedError

    def predict(self, test):
        raise NotImplementedError


class TDLambda(Model):
    def __init__(self, alpha=0.01, lamb_da=0, epsilon=0.1):
        #super().__init__(alpha, lamb_da, epsilon)
        self.alpha, self.lamb_da, self.epsilon = alpha, lamb_da, epsilon
        self.calculated_w = np.array([0., 0.5, 0.5, 0.5, 0.5, 0.5, 1.])
        self.ideal_w = np.array([0., 1. / 6, 2. / 6, 3. / 6, 4. / 6, 5. / 6, 1.])

    def fit(self, train):
        prev_w = np.array([float('inf')] * 7)
        while np.linalg.norm(prev_w - self.calculated_w) > self.epsilon:
            prev_w = deepcopy(self.calculated_w)
            dW = np.zeros_like(self.calculated_w)
            for x in train:
                for t in range(len(x)):
                    lambda_sum = np.zeros_like(self.calculated_w)
                    for k in range(1, t + 1):
                        lambda_sum += (self.lamb_da ** (t - k)) * x[k]
                    x_prime = x[t + 1] if t + 1 < len(x) else x[t]
                    dW += self.alpha * (np.dot(x_prime, np.array(self.calculated_w).T) - np.dot(x[t], np.array(self.calculated_w).T)) * lambda_sum
            self.calculated_w += dW


    def fit1(self, train):
        for x in train:
            for t in range(len(x)):
                lambda_sum = np.zeros_like(self.calculated_w)
                for k in range(1, t + 1):
                    lambda_sum += (self.lamb_da ** (t - k)) * x[k]
                x_prime = x[t + 1] if t + 1 < len(x) else x[t]
                self.calculated_w += self.alpha * (np.dot(x_prime, np.array(self.calculated_w).T) - np.dot(x[t], np.array(self.calculated_w).T)) * lambda_sum