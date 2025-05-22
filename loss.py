import torch
import torch.nn as nn
import torch.nn.functional as F

class ODKL_Loss(nn.Module):
    def __init__(self, epsilon=1e-8, alpha=1.2, beta=2.0):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.relu = nn.ReLU()
        self.bins = [0, 2, 4, float('inf')] 

    def _get_interval_probs(self, matrix):
        probs = []
        for i in range(len(self.bins) - 1):
            mask = (matrix > self.bins[i]) & (matrix <= self.bins[i + 1])
            probs.append(mask.float().sum())
        probs = torch.stack(probs) + self.epsilon
        return F.softmax(probs, dim=0)

    def forward(self, predict, truth):
        mask = (truth < 1).float()
        loss_mask = ((predict - truth) ** 2) * (1 - mask)
        masked_loss = (mask * (self.relu(predict) - truth) ** 2)
        loss_1 = torch.mean(loss_mask + masked_loss)
        kl_loss = 0.0
        num = truth.shape[0]
        for b in range(num):
            p_b = self._get_interval_probs(truth[b])
            p_hat_b = self._get_interval_probs(predict[b])
            kl_div = (p_b * (torch.log(p_b + self.epsilon) -
                             torch.log(p_hat_b + self.epsilon))).sum()
            kl_loss += kl_div

        return 1.2 * loss_1 + 2 * (kl_loss / num)