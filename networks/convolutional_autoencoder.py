import torch

import torch.nn as nn


class AutoEncoder1D(nn.Module):
    def __init__(self, input_size, num_steps, kernel_size=3, features_multiplier=1.5):
        """
        :param input_size: How many features in the input
        :param bottle_neck_size: The size of the latent descriptor / num of features
        :param num_steps: how many layers the encoder/decoder should have
        """
        super(AutoEncoder1D, self).__init__()

        # parameters
        self.input_size = input_size
        self.num_steps = num_steps
        self.kernel_size = kernel_size
        self.features_multiplier = features_multiplier

        self.bottle_neck = None
        self.layers = self.build_recursive(self.input_size, self.num_steps)
        self.net = nn.ModuleList(self.layers)

    def build_recursive(self, input_size, num_steps):
        if num_steps == 0:
            return []

        mid_size = int(input_size * self.features_multiplier)

        enc = [nn.Conv1d(in_channels=input_size,
                         out_channels=mid_size,
                         stride=2,
                         kernel_size=self.kernel_size,
                         padding=self.kernel_size // 2),
               nn.ReLU()]
        dec = [nn.ConvTranspose1d(in_channels=mid_size,
                                  out_channels=input_size,
                                  stride=2,
                                  kernel_size=self.kernel_size,
                                  padding=self.kernel_size // 2),
               nn.ReLU()]
        mid = self.build_recursive(mid_size, num_steps - 1)
        if num_steps == 1:
            self.bottle_neck = enc[1]

        return enc + mid + dec

    def forward(self, x, return_intermidiate=False):
        interm = None
        sizes = []
        for l in self.layers:
            if isinstance(l, nn.Conv1d):
                sizes.append(x.shape[-1])
                x = l(x)
            elif isinstance(l, nn.ConvTranspose1d):
                # x = l(x)
                x = l(x, output_size=(sizes.pop(),))
            else:
                x = l(x)
            if return_intermidiate and l == self.bottle_neck:
                interm = x

        return (x, interm) if return_intermidiate else x

    @property
    def parameters_count(self):
        return sum([p.numel() for p in self.parameters()])

if __name__ == '__main__':
    model = AutoEncoder1D(input_size=14, num_steps=4, kernel_size=3, features_multiplier=1.25)
    print(model)
    print(f"Parameters: {model.parameters_count}")

    x = torch.rand((1, 14, 10))  # batch, time, features

    print(f"Input shape: {x.shape}")
    y = model(x)
    print(f"Output shape: {y.shape}")

    y, interm = model(x, True)
    print(f"Output shape: {y.shape}")
    print(f"Intermediate shape: {interm.shape}")
