__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/07/21 11:30:00"
__description__ = " Debugging of script minimizing the entropy of curve "

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

import numpy as np

from copy import deepcopy
from torch import nn

from ..VAE_accessor import VAEAccessor

# Prepare nice latent space
n_points = 100
xy_min, xy_max = -3, 3
z_grid = torch.stack([m.flatten() for m in torch.meshgrid(2 * [torch.linspace(xy_min, xy_max, n_points)])]).t()

latent_space_entropy = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])

latent_space = latent_space_entropy.pdf(z_grid)
print(latent_space.shape)
print(latent_space)


def create_latent_space_contour(z_grid, latent_space, n_points, curves=None):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.contourf(z_grid[:, 0].reshape(n_points, n_points),
                z_grid[:, 1].reshape(n_points, n_points),
                latent_space.reshape(n_points, n_points), 40, levels=50, cmap='Greys_r',
                zorder=0)

    if curves:
        original_curve = curves[0]
        optimized_curve = curves[1]

        ax.plot(original_curve[:, 0], original_curve[:, 1], color='green', marker='o',
                markerfacecolor='blue', markersize=12)
        ax.plot(optimized_curve[:, 0], optimized_curve[:, 1], color='red', marker='o',
                markerfacecolor='blue', markersize=12)

    fig.savefig("./latent_space_entropy.png", dpi=600)


def curve_energy(curve):
    # if curve.dim() == 2: curve.unsqueeze_(0)  # BxNxd
    # with torch.no_grad():
    print("-------------")
    print(curve[:5])
    # recon = latent_space_entropy.pdf(curve)  # BxNxFxS
    final_shape = (z_grid.shape[0], -1, len(MSA.aa) + 1)
    log_p = VAEAccessor.vae.decoder(curve, c=solubility)
    probs = torch.unsqueeze(log_p, -1)
    probs = probs.view(final_shape)
    probs = torch.exp(probs)
    recon =
    print(recon[:5])
    return torch.tensor(recon, requires_grad=True)
    # x = recon[:, :-1, :, :]
    # y = recon[:, 1:, :, :]  # Bx(N-1)xFxS
    # dt = torch.norm(curve[:, :-1, :] - curve[:, 1:, :], p=2, dim=-1)  # Bx(N-1)
    # energy = (1 - (x * y).sum(dim=2)).sum(dim=-1)  # Bx(N-1)
    # return 2 * (energy * dt).sum(dim=-1)


def numeric_curve_optimizer_tmp(curve):
    optimizer = torch.optim.Adam(curve.parameters(), lr=1e-2)
    alpha = torch.linspace(0, 1, 50).reshape((-1, 1))
    best_curve, best_loss = deepcopy(curve), float('inf')
    criterion = torch.nn.MSELoss(reduction='sum')

    for i in range(10):
        optimizer.zero_grad()
        pred_entropy = curve(torch.tensor([1.0, 2.0], dtype=torch.float64)).sum()
        print(pred_entropy)
        print(torch.zeros(pred_entropy.shape))
        loss = criterion(pred_entropy, torch.zeros(pred_entropy.shape, dtype=torch.float64))
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        loss.backward()
        optimizer.step()

        # grad_size = torch.max(torch.abs(curve.parameters.grad))
        # if grad_size < 1e-3:
        #     break
        if loss.item() < best_loss:
            best_curve = deepcopy(curve)
            best_loss = loss.item()
        curve.update_curve(pred_entropy)

    return best_curve


class CurveWrapper(nn.Module):
    def __init__(self, start, end, n_units=100):
        super(CurveWrapper, self).__init__()
        self.original_curve = np.linspace(start, end, n_units + 1, endpoint=True)
        self.curve = nn.Parameter(torch.from_numpy(np.linspace(start, end, n_units + 1, endpoint=True)),
                                  requires_grad=True)
        # self.a = nn.Parameter(torch.randn(1), requires_grad=True)
        # self.b = nn.Parameter(torch.randn(1), requires_grad=True)
        # self.c = nn.Parameter(torch.randn(1), requires_grad=True)
        # self.d = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        # return self.a + self.b * x + self.c ** 2 * x + self.d ** 3 * x
        with torch.no_grad():
            probs = latent_space_entropy.pdf(self.curve)
            with torch.enable_grad():
                return probs

    def get_original_optimized_curves(self):
        return [self.original_curve, self.curve.detach().numpy()]

    def update_curve(self, curve):
        self.curve = curve

def numeric_curve_optimizer( curve):
    optimizer = torch.optim.Adam([curve.parameters], lr=1e-2)
    alpha = torch.linspace(0, 1, 50).reshape((-1, 1))
    best_curve, best_loss = deepcopy(curve), float('inf')
    for i in range(10):
        optimizer.zero_grad()
        print(curve(alpha))
        print("##############################")
        loss = curve_energy(curve(alpha)).sum()
        loss.backward()
        optimizer.step()
        grad_size = torch.max(torch.abs(curve.parameters.grad))
        if grad_size < 1e-3:
            break
        if loss.item() < best_loss:
            best_curve = deepcopy(curve)
            best_loss = loss.item()

    return best_curve

# curve = CurveWrapper([-0.5, -0.5], [1.0, 1.0])
x1 = torch.tensor([-0.5, 0.0])
x2 = torch.tensor([1.0, 0.0])
curve = DiscreteCurve(x1, x2)
optimized_curve = numeric_curve_optimizer(curve)

# create_latent_space_contour(z_grid, latent_space, n_points, curve.get_original_optimized_curves())
# dis = DiscreteCurve(x1, x2)
# print(dis)