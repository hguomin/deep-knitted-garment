# Guomin: the fusion layer for garment prediction

import torch
import torch.nn as nn

class ClothLinearFusion(nn.Module):
    def __init__(self, cloth_channels, body_channels):
        super().__init__()
        self.f = nn.Parameter(torch.FloatTensor(cloth_channels, body_channels))

    def reset_parameters(self, cloth_channels, body_channels):
        w_stdv = 1.0 / (cloth_channels * body_channels)
        self.f.data.uniform_(-w_stdv, w_stdv)

    def forward(self, cloth_latent, body_latent):
        batch = cloth_latent.shape[0]
        c_sum = cloth_latent.sum(axis=1)
        c_sum = c_sum.reshape(batch, -1, 1)
        cc = c_sum.detach().numpy()
        func = self.f[None, :] * c_sum
        body_latent = body_latent.reshape(batch, -1, 1)
        result = torch.matmul(func, body_latent).reshape(batch, -1)
        r = result.detach().numpy()
        return result

if __name__ == '__main__':
    f = ClothLinearFusion(10, 6)
    c = torch.arange(10, dtype=torch.float32)
    c = c.unsqueeze(0)
    c = c.expand(16, -1)
    cc = c.numpy()
    b = torch.arange(10, 16, dtype=torch.float32)
    b = b[None, :]
    b = b.expand(16, -1)
    bb = b.numpy()
    r = f(c, b)