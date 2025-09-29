import math
from statistics import NormalDist

import numpy as np
import torch

class MVEDMLoss:
    """Loss in the paper "Elucidating the Design Space of Diffusion-Based
    Generative Models" (EDM)."""

    def __init__(self, sigma_method, p_mean=-1.2, p_std=1.2, sigma_data=0.5, cam_keys=None):
        super().__init__()
        self.sde = EDMSDE(sigma_method, p_mean, p_std)
        self.scaling = EDMScaling(sigma_data)
        self.sigma_data = sigma_data
        self.cam_keys = cam_keys


    def add_noise(self, MVlatents, batch_size):
        # 每个view共享同样程度的噪声
        sigma = self.sde.sample_sigma(batch_size).reshape((batch_size, 1))
        sigma = sigma.to(MVlatents[self.cam_keys[0]].device, MVlatents[self.cam_keys[0]].dtype)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        self.sigma = sigma
        self.c_skip = c_skip
        self.c_out = c_out

        self.latents = {}
        self.noisy_latents = {}
        input_noisy_latents = {}
        for key in self.cam_keys:
            latents = MVlatents[key]  # (B, C, T, H, W)
            ori_shape = latents.shape
            latents = latents.reshape((batch_size, -1))
            sigma = sigma.to(latents.device, latents.dtype)
            # add noise
            noise = torch.randn_like(latents)  # TODO 目前每个view不一样的噪声来源，但是噪声叠加程度相同
            noisy_latents = latents + noise * sigma
            input_noisy_latents[key] = noisy_latents * c_in
            input_noisy_latents[key] = input_noisy_latents[key].reshape(ori_shape)
            self.latents[key] = latents
            self.noisy_latents[key] = noisy_latents

        return input_noisy_latents, c_noise.reshape(-1)
    
    def denoise(self, pred_latents):
        denoised_latents = {}
        for key in self.cam_keys:
            denoised_latents[key] = self.c_skip * self.noisy_latents[key] + self.c_out * pred_latents[key].reshape(self.latents[key].shape)
            denoised_latents[key] = denoised_latents[key].reshape(pred_latents[key].shape)
        return denoised_latents

    def get_loss_weight(self):
        return (self.sigma**2 + self.sigma_data**2) / (self.sigma * self.sigma_data) ** 2

    def compute_loss(self, denoised_latents):
        loss_weight = self.get_loss_weight()
        total_loss = 0
        for key in self.cam_keys:
            loss = loss_weight * (denoised_latents[key].reshape(self.latents[key].shape) - self.latents[key]) ** 2
            loss = torch.mean(loss, dim=1)
            total_loss += loss
        return total_loss



class EDMLoss:
    """Loss in the paper "Elucidating the Design Space of Diffusion-Based
    Generative Models" (EDM)."""

    def __init__(self, sigma_method, p_mean=-1.2, p_std=1.2, sigma_data=0.5):
        super().__init__()
        self.sde = EDMSDE(sigma_method, p_mean, p_std)
        self.scaling = EDMScaling(sigma_data)
        self.sigma_data = sigma_data

    def add_noise(self, latents, noise=None):
        # latents: (B, C, T, H, W)
        ori_shape = latents.shape
        batch_size = latents.shape[0]
        latents = latents.reshape((batch_size, -1))
        sigma = self.sde.sample_sigma(batch_size).reshape((batch_size, 1))
        sigma = sigma.to(latents.device, latents.dtype)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        # add noise
        if noise is None:
            noise = torch.randn_like(latents)
        noisy_latents = latents + noise * sigma
        input_noisy_latents = noisy_latents * c_in
        self.sigma = sigma
        self.c_skip = c_skip
        self.c_out = c_out
        self.latents = latents
        self.noisy_latents = noisy_latents
        return input_noisy_latents.reshape(ori_shape), c_noise.reshape(-1)

    def denoise(self, pred_latents):
        denoised_latents = self.c_skip * self.noisy_latents + self.c_out * pred_latents.reshape(self.latents.shape)
        return denoised_latents.reshape(pred_latents.shape)

    def get_loss_weight(self):
        return (self.sigma**2 + self.sigma_data**2) / (self.sigma * self.sigma_data) ** 2

    def compute_loss(self, denoised_latents):
        loss_weight = self.get_loss_weight()
        loss = loss_weight * (denoised_latents.reshape(self.latents.shape) - self.latents) ** 2
        loss = torch.mean(loss, dim=1)
        return loss


class EDMScaling:
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise


class EDMSDE:
    def __init__(self, sigma_method, p_mean=-1.2, p_std=1.2, seed=-1):
        self.sigma_method = sigma_method
        self.p_mean = p_mean
        self.p_std = p_std
        self.gaussian_dist = NormalDist(mu=p_mean, sigma=p_std)
        if seed > 0:
            self.generator = np.random.default_rng(seed)
        else:
            self.generator = np.random

    def sample_sigma(self, batch_size):
        if self.sigma_method == 1:
            log_sigma = torch.normal(mean=self.p_mean, std=self.p_std, size=(batch_size,))
            sigma = torch.exp(log_sigma)
        elif self.sigma_method == 2:
            sigma = rand_cosine_interpolated(shape=(batch_size,))
        elif self.sigma_method == 3:
            cdf_vals = self.generator.uniform(size=(batch_size,))
            samples_interval_gaussian = [self.gaussian_dist.inv_cdf(cdf_val) for cdf_val in cdf_vals]
            log_sigma = torch.tensor(samples_interval_gaussian)
            sigma = torch.exp(log_sigma)
        else:
            assert False
        return sigma


# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f'groups must be positive, got {groups}')
    if group < 0 or group >= groups:
        raise ValueError(f'group must be in [0, {groups})')
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(
    shape,
    image_d=64,
    noise_d_low=32,
    noise_d_high=64,
    sigma_data=0.5,
    min_value=0.002,
    max_value=700,
    device='cpu',
    dtype=torch.float32,
):
    """Draws samples from an interpolated cosine timestep distribution (from
    simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(shape, group=0, groups=1, dtype=dtype, device=device)
    logsnr = logsnr_schedule_cosine_interpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data
