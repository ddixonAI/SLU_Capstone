import torch
import torch.nn as nn


class conv_att_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.depthwise = nn.Conv2d(in_c, out_c, kernel_size=3, padding='valid', group=in_c)
        self.ln1 = nn.LayerNorm(in_c)
        self.flatten = torch.flatten(in_c, start_dim=1)

    def forward(self, inputs):

        x = self.depthwise(inputs)
        x = self.ln1(x)
        x = self.flatten(x)


class fct_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.ln1 = nn.LayerNorm()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.mp1 = nn.MaxPool2d((2,2))

    def forward(self, inputs):

        x = self.ln1(inputs)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.mp1(x)

        return x

