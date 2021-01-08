import torch
import torch.nn as nn


def gp(critic, real, fake, label, device="cpu"):

    BATCH_SIZE, C, H, W = real.size()
    epsilon = torch.rand(BATCH_SIZE, 1, 1, 1).repeat(1, C, H, W).to(device)
    try:
        interpolated = real * epsilon + fake * (1 - epsilon)
    except Exception as e:
        print(e)
        print(real.size())
        print(fake.size())
        print(epsilon.size())
        exit(1)
    mixed_score = critic(interpolated, label)
    gradient = torch.autograd.grad(inputs=interpolated, outputs=mixed_score, grad_outputs=torch.ones_like(mixed_score),
                                   create_graph=True, retain_graph=True)[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penality = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penality
