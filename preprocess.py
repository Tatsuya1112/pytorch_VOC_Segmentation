import numpy as np
import torch

class Crop_128:
	def __call__(self, x):
		W, H = x.size
		D = min(W, H)
		x = x.crop(((W-D)/2, (H-D)/2, (W+D)/2, (H+D)/2))
		x = x.resize((128, 128))
		return x

class Target_Preprocess:
	def __call__(self, x):
		x = np.array(x)
		return torch.tensor(x, dtype=torch.long)