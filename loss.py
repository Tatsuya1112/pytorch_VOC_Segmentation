import torch.nn as nn
import torch.nn.functional as F
import torch

class CrossEntropyLoss_aux(nn.Module):

	def __init__(self, aux_weight=0.4):
		super().__init__()
		self.aux_weight = aux_weight

	def forward(self, outputs, targets):
		loss = F.cross_entropy(outputs["out"], targets, reduction="mean", ignore_index=255)
		loss_aux = F.cross_entropy(outputs["aux"], targets, reduction="mean", ignore_index=255)

		return loss+loss_aux
