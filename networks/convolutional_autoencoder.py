import torch

import torch.nn as nn


class AutoEncoder1D(nn.Module):
    def __init__(self, input_size, num_steps, kernel_size=3):
        """
        :param input_size: How many features in the input
        :param bottle_neck_size: The size of the latent descriptor / num of features
        :param num_steps: how many layers the encoder/decoder should have
        """
        super(AutoEncoder1D, self).__init__()

        self.kernel_size = kernel_size
        self.bottle_neck = None
        self.layers = self.build_recursive(input_size, num_steps)
        self.net = nn.ModuleList(self.layers)

    def build_recursive(self, input_size, num_steps):
        if num_steps == 0:
            return []

        # diff = (input_size - self.bottle_neck_size) // num_steps
        # out_size = input_size - diff
        enc = [nn.Conv1d(in_channels=input_size,
                         out_channels=input_size * 2,
                         stride=2,
                         kernel_size=self.kernel_size,
                         padding=2),
               nn.ReLU()]
        dec = [nn.ConvTranspose1d(in_channels=input_size * 2,
                                  out_channels=input_size,
                                  stride=2,
                                  kernel_size=self.kernel_size),
               nn.ReLU()]
        mid = self.build_recursive(input_size * 2, num_steps - 1)
        if num_steps - 1 == 1:
            self.bottle_neck = mid[0]

        return enc + mid + dec

    def forward(self, x, return_intermidiate=False):
        interm = None
        sizes = []
        for l in self.layers:
            if isinstance(l, nn.Conv1d):
                sizes.append(x.shape[-1])
                x = l(x)
            elif isinstance(l, nn.ConvTranspose1d):
                x = l(x, output_size=(sizes.pop(),))
            else:
                x = l(x)
            if return_intermidiate and l == self.bottle_neck:
                interm = x

        return (x, interm) if return_intermidiate else x


if __name__ == '__main__':
    model = AutoEncoder1D(input_size=14, num_steps=2, kernel_size=5)
    print(model)

    x = torch.rand((1, 14, 20))  # batch, time, features
    y = model(x)
    print(y.shape)

    y, interm = model(x, True)
    print(y.shape)
    print(interm.shape)
