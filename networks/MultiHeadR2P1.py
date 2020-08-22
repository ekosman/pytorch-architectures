import torch
import torch.nn as nn
import torchvision
import math
import numpy as np


class Identity(nn.Module):
	def forward(self, x):
		return x


class Head(nn.Module):
	def __init__(self, input_size, n_classes, time_window, n_steps, kernel_size):
		super(Head, self).__init__()
		self.stride = math.ceil((input_size / n_classes) ** (1 / n_steps))
		self.layers = []
		channels_steps = np.linspace(1, time_window, n_steps + 1, dtype=int)
		for _, in_channels, out_channels in zip(range(n_steps), channels_steps[:-1], channels_steps[1:]):
			self.layers.append(nn.Conv1d(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=kernel_size,
				padding=(kernel_size - self.stride) // 2,
				stride=self.stride
			))
			self.layers.append(nn.ReLU())

		self.net = nn.Sequential(*self.layers)

	def forward(self, x):
		return self.net(x)


class MultiHeadR2P1(nn.Module):
	def __init__(self):
		super(MultiHeadR2P1, self).__init__()
		self.backbone = torchvision.models.video.r2plus1d_18()
		self.backbone = nn.ModuleList([
			self.backbone.stem,
			self.backbone.layer1,
			self.backbone.layer2,
			self.backbone.layer3,
			# self.backbone.layer4,
		]
		)
		self.avg = nn.AdaptiveAvgPool2d(1)
		# self.backbone.stem = Identity()
		self.heads = nn.ModuleList([
			Head(input_size=512, n_classes=10, time_window=20, n_steps=3, kernel_size=5),
			Head(input_size=512, n_classes=20, time_window=5, n_steps=3, kernel_size=5)
		])

	def forward(self, x):

		for l in self.backbone:
			x = l(x)

		x = torch.transpose(x, 1, 2)
		batch_size = x.shape[0]
		time_window = x.shape[1]
		x = x.reshape(-1, *x.shape[-3:])
		x = self.avg(x)
		x = x.reshape(batch_size, time_window, -1)
		ys = [head(x) for head in self.heads]
		return tuple(ys)


if __name__ == '__main__':
	model = MultiHeadR2P1()
	x = torch.randn(2, 3, 37, 112, 112)
	y = model(x)