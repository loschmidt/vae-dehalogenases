__author__ = "Pavel Kohout <xkohou15@vutbr.cz>"
__date__ = "2022/08/20 14:57:00"
__description__ = "Using scripts from  " \
                  "https://doi.org/10.1038/s41467-022-29443-w"


from torch import nn
from sklearn.cluster import KMeans

import torch


# Entropy network
class TranslatedSigmoid(nn.Module):
    def __init__(self):
        super(TranslatedSigmoid, self).__init__()
        self.beta = nn.Parameter(torch.tensor([-3.5]))

    def forward(self, x):
        beta = torch.nn.functional.softplus(self.beta)
        alpha = -beta * 6.9077542789816375
        return torch.sigmoid((x + alpha) / beta)


class DistNet(nn.Module):
    def __init__(self, dim, num_points):
        super().__init__()
        self.num_points = num_points
        self.points = nn.Parameter(torch.randn(num_points, dim),
                                   requires_grad=False)
        self.trans = TranslatedSigmoid()
        self.initialized = False

    def __dist2__(self, x):
        t1 = (x ** 2).sum(-1, keepdim=True)
        t2 = torch.transpose((self.points ** 2).sum(-1, keepdim=True), -1, -2)
        t3 = 2.0 * torch.matmul(x, torch.transpose(self.points, -1, -2))
        return (t1 + t2 - t3).clamp(min=0.0)

    def forward(self, x):
        with torch.no_grad():  # To prevent backpropping back through to the dist points
            D2 = self.__dist2__(x)  # |x|-by-|points|
            min_d = D2.min(dim=-1)[0]  # smallest distance to clusters
            return self.trans(min_d)

    def kmeans_initializer(self, embeddings):
        km = KMeans(n_clusters=self.num_points).fit(embeddings)
        self.points.data = torch.tensor(km.cluster_centers_,
                                        device=self.points.device)
        self.initialized = True
