"""
All things related to centerloss.

The following piece of code for Center Loss has been pulled and modified based on the code from the GitHub
Repo: https://github.com/KaiyangZhou/pytorch-center-loss

Reference: Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
"""
__author__ = 'ryanquinnnelson'

import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """
    Defines a centerloss loss function.
    """

    def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
        """

        Args:
            num_classes (int): number of classes
            feat_dim (int): number of features
            device (torch.device): device on which model is being run
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """

        UserWarning: This overload of addmm_ is deprecated:
        addmm_(Number beta, Number alpha, Tensor mat1, Tensor mat2)
        Consider using one of the following signatures instead:
        addmm_(Tensor mat1, Tensor mat2, *, Number beta, Number alpha)

        Args:
            x (Tensor): feature matrix with shape (batch_size, feat_dim)
            labels (Tensor): ground truth labels with shape (batch_size)

        Returns:

        """

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())  # see UserWarning above

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss
