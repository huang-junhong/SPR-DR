import torch
import torch.nn as nn

from torch.autograd import Function


class ChannelDropoutFunc(Function):
    @staticmethod
    def forward(ctx: torch.Any, input:torch.Tensor, dropout_channel:list):
        
        mask = torch.ones_like(input)

        mask[:, dropout_channel, :, :] = 0

        output = input * mask

        ctx.mask = mask

        return output

    @staticmethod
    def backward(ctx: torch.Any, grad_outputs: torch.Any):
        grad = grad_outputs.clone()
        grad = grad * ctx.mask

        return grad, None
    

class ChannelDropout(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, input, dropout_channel):
        return ChannelDropoutFunc.apply(input, dropout_channel)
