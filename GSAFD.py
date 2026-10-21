import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

def tm_basis_grid(poles, n_grid):
    a = poles.unsqueeze(1)                                        # [N, 1]
    real_dtype = torch.float64 if poles.dtype == torch.complex128 else torch.float32
    j = torch.arange(n_grid, device=poles.device, dtype=real_dtype)
    z = torch.polar(torch.ones_like(j), 2 * math.pi * j / n_grid).unsqueeze(0)  # [1, M]

    denom = 1 - a.conj() * z
    gamma = torch.sqrt(1 - a.abs() ** 2 + 1e-12).to(poles.dtype)

    atoms = gamma / denom                       # e_{a_n}(z_j)
    blaschke = (z - a) / denom                  # phi_{a_n}(z_j)

    partial = torch.cumprod(blaschke, dim=0)                      # prod_{k<=n}
    partial = torch.cat([torch.ones_like(partial[:1]),            # shift to prod_{k<n}
                         partial[:-1]], dim=0)
    return atoms * partial                                        # [N, M]


class AFDDecompositionLayer(nn.Module):
    def __init__(self, n_levels, patch_dim, max_radius=0.85):
        super().__init__()
        if n_levels > patch_dim // 2:
            warnings.warn(
                f"AFDDecompositionLayer: n_levels={n_levels} exceeds "
                f"patch_dim//2={patch_dim // 2}. Monitor gram_cond() and "
                f"gram_error(): sampling quality depends on the pole "
                f"configuration as well as the number of atoms.",
                RuntimeWarning, stacklevel=2)
        self.n_levels = n_levels
        self.patch_dim = patch_dim
        self.max_radius = max_radius

        ratio = (init_poles.abs() / max_radius).clamp(1e-4, 1 - 1e-4)
        self.nu = nn.Parameter(torch.log(-torch.log(ratio)))
        self.theta = nn.Parameter(init_poles.angle())

    @staticmethod
    def _init_poles(n, r_min=0.12, r_max=0.76):
        n_r = max(1, int(round(math.sqrt(n / 2.0))))
        n_theta = int(math.ceil(n / n_r))
        
        r = torch.full((n_r,), 1e-4).repeat_interleave(n_theta)[:n]
        
        theta = (torch.arange(n_theta, dtype=torch.float32) * math.pi / n_theta)
        theta = theta.repeat(n_r)[:n]
        
        return torch.polar(r.contiguous(), theta.contiguous())

    def poles(self):
        r = self.max_radius * torch.exp(-torch.exp(self.nu))
        return torch.polar(r, self.theta)

    def build_basis_matrix(self):
        a = self.poles()
        grid = tm_basis_grid(a, self.patch_dim)
        b = grid / math.sqrt(self.patch_dim)
        g = b @ b.conj().transpose(0, 1)
        n = g.shape[0]
        eye = torch.eye(n, dtype=g.dtype, device=g.device)

        L, info = torch.linalg.cholesky_ex(g + 1e-4 * eye)
        if bool(info.any()):
            raise RuntimeError(
                "AFDDecompositionLayer: regularized Cholesky factorization "
                "failed. Check pole values and sampled Gram conditioning."
            )
        return torch.linalg.solve_triangular(L, b, upper=False)

    @torch.no_grad()
    def gram_cond(self):
        a = self.poles()
        grid = tm_basis_grid(a, self.patch_dim)
        b = (grid / math.sqrt(self.patch_dim)).to(torch.complex128)
        return torch.linalg.cond(b @ b.conj().transpose(0, 1))

    def differentiable_hilbert(self, x_real):
        n = x_real.shape[-1]
        X_fft = torch.fft.fft(x_real, n=n, dim=-1)
        h = torch.zeros(n, device=x_real.device, dtype=x_real.dtype)
        if n > 0 and n % 2 == 0:
            h[0] = h[n // 2] = 1; h[1:n // 2] = 2
        elif n > 0:
            h[0] = 1; h[1:(n + 1) // 2] = 2
        return torch.fft.ifft(X_fft * h, n=n, dim=-1)

    @torch.no_grad()
    def gram_error(self):
        b = self.build_basis_matrix().to(torch.complex128)
        g = b @ b.conj().transpose(0, 1)
        return (g - torch.eye(b.shape[0], dtype=g.dtype, device=g.device)).abs().max()

    def forward(self, patches_flat, return_coeffs=False):
        if patches_flat.shape[-1] != self.patch_dim:
            raise ValueError(
                f"AFDDecompositionLayer expects signals of length "
                f"{self.patch_dim}, got {patches_flat.shape[-1]}. "
                f"(patch_dim must be C * patch_size * patch_size.)")

        B_n = self.build_basis_matrix()                       # (Atoms, Pixels)

        patches_analytic = self.differentiable_hilbert(patches_flat).to(B_n.dtype)

        c_n = patches_analytic @ B_n.conj().transpose(0, 1)   # (Batch*Patches, Atoms)

        if return_coeffs:
            return c_n, B_n

        return (c_n @ B_n).real


class PatchwiseAFDDecomposer(nn.Module):
    def __init__(self, n_levels, patch_size, stride, max_radius=0.85):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.afd_decomposer = AFDDecompositionLayer(
            n_levels, patch_size * patch_size,
            max_radius=0.85)

        hann_1d = torch.hann_window(patch_size + 2, periodic=False)[1:-1]
        hann_2d = torch.outer(hann_1d, hann_1d)
        self.register_buffer('window', hann_2d.view(1, 1, patch_size, patch_size))
        self._normalizer_cache = {}

    def forward_features(self, x):
        B, C, H, W = x.shape
        self.original_size = (H, W)
        
        # 1. Pad
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        else:
            x_padded = x
        self.pad_info = (pad_h, pad_w)
        B, C, H_pad, W_pad = x_padded.shape

        # 2. Unfold
        patches = F.unfold(x_padded, kernel_size=self.patch_size, stride=self.stride)
        patches_flat = patches.permute(0, 2, 1).contiguous().view(-1, C * self.patch_size * self.patch_size)

        # 3. Mean Removal
        patch_mean = patches_flat.mean(dim=1, keepdim=True)
        patches_centered = patches_flat - patch_mean

        # 4. Snake Scan
        patches_2d = patches_centered.view(-1, C, self.patch_size, self.patch_size)
        patches_snake = patches_2d.clone()
        patches_snake[:, :, 1::2, :] = torch.flip(patches_snake[:, :, 1::2, :], dims=[-1])
        patches_snake_flat = patches_snake.view(-1, C * self.patch_size * self.patch_size)

        # 5. Get Coefficients
        c_n, B_n = self.afd_decomposer(patches_snake_flat, return_coeffs=True)
        
        return c_n, B_n, patch_mean, (B, C, H_pad, W_pad)

    def _get_normalizer(self, B, C, H_pad, W_pad, device, dtype):
        key = (B, C, H_pad, W_pad, device, dtype)
        cached = self._normalizer_cache.get(key)
        if cached is not None:
            return cached

        ones = torch.ones((B, C, H_pad, W_pad), device=device, dtype=dtype)
        ones_patches = F.unfold(ones, kernel_size=self.patch_size, stride=self.stride)
        ones_patches = ones_patches.view(B, C, self.patch_size, self.patch_size, -1)
        ones_patches = ones_patches * self.window.unsqueeze(-1).to(device=device, dtype=dtype)
        ones_patches = ones_patches.view(B, -1, ones_patches.shape[-1])

        normalizer = F.fold(ones_patches, output_size=(H_pad, W_pad),
                            kernel_size=self.patch_size, stride=self.stride)
        normalizer = normalizer.clamp_min(1e-5)
        self._normalizer_cache[key] = normalizer
        return normalizer

    def reconstruct_from_coeffs(self, c_n, B_n, patch_mean, shape_info):
        B, C, H_pad, W_pad = shape_info
        
        # 1. Reconstruct Patches (Snake Domain)
        # (N, Atoms) @ (Atoms, Pixels) -> (N, Pixels)
        reconstructed_snake = (c_n @ B_n).real
        
        # 2. Reverse Snake
        patches_snake_2d = reconstructed_snake.view(-1, C, self.patch_size, self.patch_size)
        patches_2d = patches_snake_2d.clone()
        patches_2d[:, :, 1::2, :] = torch.flip(patches_2d[:, :, 1::2, :], dims=[-1])
        
        # 3. Add Mean
        # Reshape mean for broadcasting: (N, 1) -> (N, C, 1, 1)
        mean_view = patch_mean.view(-1, C, 1, 1)
        patches_restored = patches_2d + mean_view

        patches_weighted = patches_restored * self.window

        # 4. Prepare for Fold
        # Reshape to (B, C*Px*Px, N_Patches)
        num_patches = patches_weighted.shape[0] // B
        patches_flat_weighted = patches_weighted.view(B, num_patches, -1).permute(0, 2, 1)
        
        output_size = (H_pad, W_pad)
        img_folded = F.fold(patches_flat_weighted, output_size=output_size, kernel_size=self.patch_size, stride=self.stride)

        # 5. Normalize (Windowed Divisor) -> weighted overlap-add.
        normalizer = self._get_normalizer(1, C, H_pad, W_pad, img_folded.device, img_folded.dtype)

        img_folded = img_folded / normalizer

        # 6. Crop
        original_h, original_w = self.original_size
        pad_h, pad_w = self.pad_info
        if pad_h > 0 or pad_w > 0:
            img_folded = img_folded[:, :, :original_h, :original_w]
            
        return img_folded


class LearnableCoeffFusion(nn.Module):
    def __init__(self, num_atoms=32):
        super(LearnableCoeffFusion, self).__init__()
        # Use a simple FC to decide weights based on magnitude
        self.fc = nn.Sequential(
            nn.Linear(num_atoms * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_atoms),
            nn.Sigmoid()
        )

    def forward(self, c_ir, c_vi):
        # 1. Decompose Complex Numbers
        mag_ir = torch.abs(c_ir)
        mag_vi = torch.abs(c_vi)

        # 2. Decide Fusion Weights (Based on Magnitudes)
        cat_feat = torch.cat([mag_ir, mag_vi], dim=1)
        fusion_map = self.fc(cat_feat)

        # 3. BLEND THE COMPLEX COEFFICIENTS
        c_blend = fusion_map * c_ir + (1 - fusion_map) * c_vi

        target_mag = torch.max(mag_ir, mag_vi)
        current_mag = torch.abs(c_blend) + 1e-6
        c_fused = c_blend * (target_mag / current_mag)

        return c_fused, fusion_map


class AFD_NET(nn.Module):
    def __init__(self, s, n, channel, stride, num_block, fusion_type):
        super(AFD_NET, self).__init__()
        patch_stride = stride
        n_levels_baseline = n

        patch_size = 8
 
        self.get_ls1 = PatchwiseAFDDecomposer(n_levels=n_levels_baseline, patch_size=patch_size,
                                              stride=patch_stride, max_radius=0.85)
        self.get_ls2 = PatchwiseAFDDecomposer(n_levels=n_levels_baseline, patch_size=patch_size,
                                              stride=patch_stride, max_radius=0.85)
        # Share the basis generator
        self.get_ls2.afd_decomposer = self.get_ls1.afd_decomposer

        self.coeff_fusion = LearnableCoeffFusion(num_atoms=n_levels_baseline)

    def forward(self, x, y):
        c_ir, B_n_ir, mean_ir, shape_info = self.get_ls1.forward_features(x)
        c_vi, _, mean_vi, _ = self.get_ls2.forward_features(y)

        c_fused, fusion_map = self.coeff_fusion(c_ir, c_vi)


        patch_weight = torch.mean(fusion_map, dim=1, keepdim=True)
        mean_network = patch_weight * mean_ir + (1 - patch_weight) * mean_vi
        target_mean_mag = torch.max(torch.abs(mean_ir), torch.abs(mean_vi))

        current_mean_mag = torch.abs(mean_network) + 1e-6
        scale_dc = target_mean_mag / current_mean_mag
        mean_boosted = mean_network * scale_dc
        
        certainty = torch.abs(2.0 * patch_weight - 1.0)
        dynamic_beta = 0.9 * certainty  
        
        mean_fused = dynamic_beta * mean_boosted + (1 - dynamic_beta) * mean_network

        img_fused = self.get_ls1.reconstruct_from_coeffs(c_fused, B_n_ir, mean_fused, shape_info)

        out = {
            'fuse': img_fused,
            'rec_ir': self.get_ls1.reconstruct_from_coeffs(c_ir, B_n_ir, mean_ir, shape_info),
            'rec_vi': self.get_ls1.reconstruct_from_coeffs(c_vi, B_n_ir, mean_vi, shape_info)
        }
        return out
        
class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        b, c, h, w = X.shape
        if c == 1:
            X = X.repeat(1, 3, 1, 1)
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h
        return [relu1_2, relu2_2, relu3_3, relu4_3]
