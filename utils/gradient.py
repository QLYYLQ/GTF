import torch
from torch import nn


def gradient(input):
    """
    求图像梯度, sobel算子
    :param input:
    :return:
    """

    filter1 = nn.Conv2d(
        kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1
    )
    filter2 = nn.Conv2d(
        kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1
    )
    filter1.weight.data = (
        torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        .reshape(1, 1, 3, 3)
        .cuda()
    )
    filter2.weight.data = (
        torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        .reshape(1, 1, 3, 3)
        .cuda()
    )

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient
