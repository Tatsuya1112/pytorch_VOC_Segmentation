import torch
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def init_weights(m):
    torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)

def lambda_epoch(epoch):
    max_epoch = 30
    return math.pow((1-epoch/max_epoch),0.9)

