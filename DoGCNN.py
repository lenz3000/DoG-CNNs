'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.distributions import normal, beta, uniform
from torch.autograd import Variable
import matplotlib.pyplot as plt

from datetime import datetime


class LeNet(nn.Module):
    def __init__(self, in_channels=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def show_filters(self, num=5):
        filters_1 = self.conv1.weight
        filters_2 = self.conv2.weight
        fig, axes = plt.subplots(nrows=2, ncols=num, sharey='row')
        filters_1 = filters_1.reshape((-1, 5, 5))
        filters_2 = filters_2.reshape((-1, 5, 5))
        max_ = torch.max(torch.max(filters_1[:num]), torch.max(filters_2[:num]))
        min_ = torch.min(torch.min(filters_1[:num]), torch.min(filters_2[:num]))
        for filter in range(num):
            axes[0, filter].imshow(filters_1[filter].detach().cpu().numpy(), vmin=min_, vmax=max_)
            im = axes[1, filter].imshow(filters_2[filter].detach().cpu().numpy(), vmin=min_, vmax=max_)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()


class DoGConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', mean_fac=1, var_fac=.5):

        super(DoGConv, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        # My initialization of the parameters is really hacky and not really useful, but it's only used once,
        # so I did not optimize it
        # First we sample the parameters for the Difference of Gaussian Filters
        uni = uniform.Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))

        # For the variance we use the Cholesky decomposition in order to enforce pos. def.
        print('Initializing')
        self.means = nn.Parameter(mean_fac * uni.sample((out_channels, in_channels, 2, 1, 2))[..., 0])
        var_chol = var_fac * uni.sample((out_channels, in_channels, 2, 2, 2))[..., 0]
        var_chol[var_chol == 0] += 1E-1
        self.var_chol = nn.Parameter(torch.tril(var_chol))
        self.ratios = nn.Parameter(((1 + 1E-8 + (1 - 2E-8) * uni.sample((out_channels, in_channels,))) / 2)[..., 0])
        self.last_filters = None
        if bias:
            self.bias = nn.Parameter(uni.sample((out_channels,))[..., 0])
        else:
            self.bias = None
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        resol = np.linspace(-1, 1, self.kernel_size)
        pos = torch.tensor(np.stack(np.meshgrid(resol, resol), axis=2), dtype=torch.float32)
        self.register_buffer('flat_pos', pos.reshape((1, self.kernel_size ** 2, 2)))

    def get_filters(self):
        """
        This functions creates the kernel-weights from the two Gaussians. This is the part that slows training down
        :return: filters
        """
        # We calculate diff = x - mu
        diffs = self.flat_pos - self.means
        # Torch bmm does not accept the dimensionalities, I could have reshaped as well.
        # This prolly leads to speedup, but has worse readability
        sigmas = torch.einsum('oipze,oipde->oipzd ', self.var_chol, self.var_chol)
        determinants = sigmas[..., 0, 0] * sigmas[..., 1, 1]
        sigmas[torch.abs(determinants) < 1E-8][..., (0, 1),(0, 1)] += 1E-8
        sigma_inv = sigmas.inverse()
        gauss_exp_ = torch.einsum('oicpd,oicde -> oicpe', diffs, sigma_inv)
        gauss_exp = torch.einsum('oicpe, oicpe->oicp', gauss_exp_, diffs)
        prop_gauss = torch.exp(-.5 * gauss_exp)
        gauss = torch.einsum('oip, oipc -> oipc', 1 / torch.sqrt(determinants), prop_gauss)
        plus = torch.einsum('oi, oic -> oic', self.ratios,
                            gauss[:, :, 0, :])
        minus = torch.einsum('oi, oic -> oic', 1 - self.ratios,
                             gauss[:, :, 1, :])
        self.last_filters = (plus - minus).reshape((self.out_channels, self.in_channels,
                                                    self.kernel_size, self.kernel_size))
        if torch.isnan(self.last_filters).sum():
            print('NAn in the Filters')
            print(self.last_filters)
            raise Exception
        return self.last_filters

    def forward(self, input):
        filters = self.get_filters()
        res = F.conv2d(input, filters, bias=self.bias, stride=self.stride, padding=self.padding,
                       dilation=self.dilation, groups=self.groups)

        return res


class LeDoGNet(nn.Module):
    def __init__(self, in_channels=3, mean_fac=1, var_fac=.5):
        super(LeDoGNet, self).__init__()

        self.conv1 = DoGConv(in_channels, 6, 5, mean_fac=mean_fac, var_fac=var_fac)
        self.conv2 = DoGConv(6, 16, 5, mean_fac=mean_fac, var_fac=var_fac)
        self.norm = nn.BatchNorm1d(16 * 5 * 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.norm(out) #I added this because the gradients lead to some NAN weights in the linear layers
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        if torch.isnan(out).sum():
            print('NAN in output')
        return out

    def show_filters(self, num=5, standardized=True):
        filters_1 = self.conv1.get_filters()
        filters_2 = self.conv2.get_filters()
        fig, axes = plt.subplots(nrows=2, ncols=num, sharey='row')
        filters_1 = filters_1.reshape((-1, self.conv1.kernel_size, self.conv1.kernel_size))
        filters_2 = filters_2.reshape((-1, self.conv2.kernel_size, self.conv2.kernel_size))
        kwargs = {}
        if standardized:
            kwargs['vmax'] = torch.max(torch.max(filters_1[:num]), torch.max(filters_2[:num]))
            kwargs['vmin'] = torch.min(torch.min(filters_1[:num]), torch.min(filters_2[:num]))
        for filter in range(num):
            axes[0, filter].imshow(filters_1[filter].detach().cpu().numpy(), **kwargs)
            im = axes[1, filter].imshow(filters_2[filter].detach().cpu().numpy(), **kwargs)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()
