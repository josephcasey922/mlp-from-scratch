import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.correct_probs = None

    def forward(self, pred, label):
        self.num_classes = pred.shape[1]
        self.label = label
        self.batch_size = pred.shape[0]
        clipped_pred = np.clip(pred, 1e-12, 1)
        self.row_indices = np.arange(self.batch_size)
        self.correct_probs = clipped_pred[self.row_indices, self.label]
        loss = -np.sum(np.log(self.correct_probs))/self.batch_size
        return loss

    def backward(self):
        grad_input = np.zeros((self.batch_size, self.num_classes), dtype=np.float64)
        grad_input[self.row_indices, self.label] = (-1 / self.correct_probs)
        return grad_input / self.batch_size