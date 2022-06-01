import torch.nn as nn
import torch


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self, mask, data_range=1, K=(0.01, 0.03)):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        K1, K2 = K
        self.C1 = (K1 * data_range) ** 2
        self.C2 = (K2 * data_range) ** 2
        self.mask = mask

    def forward(self, x, y):
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        DSSIM = torch.clamp((1 - SSIM_n / SSIM_d) / 2.0, 0, 1)

        _, _, th, tw = DSSIM.size()
        _, _, sh, sw = self.mask.size()
        dh, dw = (sh-th)/2, (sw-tw)/2
        mask_c = self.mask[:, :, dh:dh+th, dw:dw+tw]
        dssim = (torch.clamp((1 - SSIM_n / SSIM_d) / 2.0, 0, 1) * mask_c).sum() / mask_c.sum()

        return dssim