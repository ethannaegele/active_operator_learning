#!/usr/bin/env python3

"""
FNO. Implementation taken and modified from
https://github.com/zongyi-li/fourier_neural_operator

MIT License

Copyright (c) 2020 Zongyi Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from dataclasses import dataclass
from al4pde.models.unc_head import GAP1dAndFC, GAP2dAndFC, FCs

@dataclass
class UncertaintyTapSpec:
    name: str
    #enabled: bool
    block_idx: int
    in_channels: int
    hidden_channels: int
    out_channels: int
    num_layers: int = 1
    make_channel_first_input: bool = False
    use_end_activation: bool = True
    detach: bool | None = None


@dataclass
class UncertaintySpec:
    enabled: bool
    taps: list[UncertaintyTapSpec]
    #aggregator: str = "sum
    detach_features: bool = False
    clamp_log_sigma: bool = False
    min_log_sigma: float | None = None
    max_log_sigma: float | None = None


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes 
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        if self.modes1 > x.size(-1) // 2 + 1:
            raise ValueError("Input of shape " + str(x.shape) + " too small for using " + str(self.modes1) + " modes")
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, 
                 num_channels, 
                 modes=16, 
                 width=64, 
                 initial_step=10, 
                 predict_delta=False,
                 uncertainty_spec: UncertaintySpec | None = None):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(initial_step * num_channels + 1, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)
        self.predict_delta = predict_delta

        self.uncertainty_spec = uncertainty_spec
        self.uncertainty_enabled = uncertainty_spec is not None and uncertainty_spec.enabled
        self.unc_heads = nn.ModuleDict()
        self.unc_taps_by_block = {}
        num_out_channels_sum = 0
        if self.uncertainty_enabled:
            for tap in uncertainty_spec.taps:
                if tap.name != "last":
                    num_out_channels_sum += tap.out_channels
            for tap in uncertainty_spec.taps:
                if tap.name != "last":
                    self.unc_heads[tap.name] = GAP1dAndFC(
                        in_channels=tap.in_channels,
                        out_channels=tap.out_channels,
                        num_layers=tap.num_layers,
                        hidden_channels=tap.hidden_channels,
                        make_channel_first_input=tap.make_channel_first_input,
                        use_end_activation=tap.use_end_activation
                        )
                    
                else:
                    self.unc_heads[tap.name] = FCs(
                                            in_channels=num_out_channels_sum,
                                            out_channels=tap.out_channels,
                                            num_layers=tap.num_layers,
                                            hidden_channels=tap.hidden_channels,
                                            use_end_activation=tap.use_end_activation # should generally be False
                                        )
                self.unc_taps_by_block.setdefault(tap.block_idx, []).append(tap)

    def _add_unc_outputs(self, unc_outputs, block_idx, x):
        if not self.uncertainty_enabled:
            return None

        taps = self.unc_taps_by_block.get(block_idx, [])
        for tap in taps:
            detach = self.uncertainty_spec.detach_features if tap.detach is None else tap.detach
            x_unc = x.detach() if detach else x

            out = self.unc_heads[tap.name](x_unc)
            unc_outputs.append(out)

    def _aggregate_unc_outputs(self, unc_outputs, block_idx):
        if len(unc_outputs) == 0:
            raise RuntimeError("return_uncertainty=True but no uncertainty taps" \
            " are enabled")

        stacked = torch.cat(unc_outputs, dim=-1)
        last_tap = self.unc_taps_by_block.get(block_idx, [])
        if len(last_tap) != 1:
            raise RuntimeError("Exactly one final uncertainty tap is required")
        last_tap = last_tap[0]
        detach = self.uncertainty_spec.detach_features if last_tap.detach is None else last_tap.detach
        stacked_unc = stacked.detach() if detach else stacked

        log_sigma = self.unc_heads[last_tap.name](stacked_unc)

        if self.uncertainty_spec.clamp_log_sigma:
            if self.uncertainty_spec.min_log_sigma is None and self.uncertainty_spec.max_log_sigma is None:
                raise ValueError("Invalid uncertainty spec. " \
                "Tried to clamp log sigma with both min_log_sigma " \
                "and max_log_sigma = None")
            log_sigma = torch.clamp(
                log_sigma,
                min=self.uncertainty_spec.min_log_sigma,
                max=self.uncertainty_spec.max_log_sigma,
            )

        return log_sigma

    def forward(self, x, grid, return_features=False, return_uncertainty=False):
        # x dim = [b, x1, t*v]
        if return_features and return_uncertainty:
            raise ValueError("""return_features=True and return_uncertainty=True
              is not supported. Uncertainty-based active learning and
              feature-based active learning are two distinct modes and should
              be used separately.""")
        features = []
        unc_outputs = []
        inp = x
        #print("input shape: ", x.shape)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        #print("shape after encoding and before padding: ", x.shape)

        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic
        
        #print("shape after padding: ", x.shape)
        self._add_unc_outputs(unc_outputs, 0, x)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        features.append(x)
        x = F.gelu(x)
        self._add_unc_outputs(unc_outputs, 1, x)
        
        #print("shape after Fourier block 0: ", x.shape)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        features.append(x)
        x = F.gelu(x)
        self._add_unc_outputs(unc_outputs, 2, x)
        
        #print("shape after Fourier block 1: ", x.shape)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        features.append(x)
        x = F.gelu(x)
        self._add_unc_outputs(unc_outputs, 3, x)
        
        #print("shape after Fourier block 2: ", x.shape)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        features.append(x)
        self._add_unc_outputs(unc_outputs, 4, x)
        
        #print("shape after Fourier block 3: ", x.shape)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)
        #print("shape immediately before decoder layer 0: ", x.shape)
        x = self.fc1(x)
        features.append(x)
        x = F.gelu(x)
        self._add_unc_outputs(unc_outputs, 5, x)
        #print("shape after decoder layer 0: ", x.shape)
        x = self.fc2(x)
        #print("shape after decoder layer 1: ", x.shape)
        if self.predict_delta:
            x = x + inp
        x = x.unsqueeze(-2)
        if return_uncertainty:
            log_sigma = self._aggregate_unc_outputs(unc_outputs, block_idx=6)
            return x, log_sigma
        if return_features:
            return x, features
        return x


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, num_channels, modes1=12, modes2=12, width=20, initial_step=10):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x, y, c)
        output: the solution of the next timestep
        output shape: (batchsize, x, y, c)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(initial_step * num_channels + 2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x, grid, return_features=False):
        # x dim = [b, x1, x2, t*v]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]  # Unpad the tensor
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        features = x
        x = self.fc2(x)
        if return_features:
            return x.unsqueeze(-2), features.unsqueeze(-2)
        return x.unsqueeze(-2)


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, num_channels, modes1=8, modes2=8, modes3=8, width=20, initial_step=10):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(initial_step * num_channels + 3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x, grid, return_features=False):
        # x dim = [b, x1, x2, x3, t*v]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        features = x
        x = self.fc2(x)
        if return_features:
            return x.unsqueeze(-2), features.unsqueeze(-2)
        return x.unsqueeze(-2)


class SpectralConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = tuple(modes)
        self.n_dims = len(self.modes)

        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.ParameterList([
            nn.Parameter(self.scale * torch.rand(in_channels, out_channels, *self.modes, dtype=torch.cfloat))
            for _ in product([0, 1], repeat=self.n_dims - 1)
        ])

    def compl_mul(self, input, weights):
        dims = "xyzt"[:self.n_dims]
        return torch.einsum(f"bi{dims},io{dims}->bo{dims}", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        sizes = tuple(x.shape[-self.n_dims:])
        ft_sizes = sizes[:-1] + (sizes[-1] // 2 + 1,)
        for mode, size in zip(self.modes[:-1], sizes[:-1]):
            if mode > size:
                raise ValueError("Input of shape " + str(x.shape) + " too small for using " + str(mode) + " modes")
        if self.modes[-1] > ft_sizes[-1]:
            raise ValueError("Input of shape " + str(x.shape) + " too small for using " + str(self.modes[-1]) + " modes")

        x_ft = torch.fft.rfftn(x, dim=list(range(-self.n_dims, 0)))
        out_ft = torch.zeros(batchsize, self.out_channels, *ft_sizes, dtype=torch.cfloat, device=x.device)

        for weight, signs in zip(self.weights, product([0, 1], repeat=self.n_dims - 1)):
            slices = [slice(None), slice(None)]
            for sign, mode in zip(signs, self.modes[:-1]):
                slices.append(slice(0, mode) if sign == 0 else slice(-mode, None))
            slices.append(slice(0, self.modes[-1]))
            slices = tuple(slices)
            out_ft[slices] = self.compl_mul(x_ft[slices], weight)

        return torch.fft.irfftn(out_ft, s=sizes, dim=list(range(-self.n_dims, 0)))


class FNOFullTrajectory(nn.Module):
    """Space-time FNO that predicts all requested trajectory frames in one pass."""

    def __init__(self, 
                 num_channels, 
                 coord_dim, 
                 modes, 
                 width=64, 
                 initial_step=1, 
                 padding=2,
                 num_physical_channels: int | None = None,
                 uncertainty_spec: UncertaintySpec | None = None):
        super(FNOFullTrajectory, self).__init__()
        self.num_channels = num_channels
        self.coord_dim = coord_dim
        self.modes = tuple(modes)
        self.width = width
        self.initial_step = initial_step
        self.padding = padding
        self.n_dims = len(self.modes)

        self.fc0 = nn.Linear(initial_step * num_channels + coord_dim, self.width)
        self.convs = nn.ModuleList([SpectralConvNd(self.width, self.width, self.modes) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Linear(self.width, self.width) for _ in range(4)])
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

        self.num_physical_channels = num_physical_channels
        self.uncertainty_spec = uncertainty_spec
        self.uncertainty_enabled = uncertainty_spec is not None and uncertainty_spec.enabled
        self.unc_heads = nn.ModuleDict()
        self.unc_taps_by_block = {}
        num_out_channels_sum = 0
        if self.uncertainty_enabled:
            for tap in uncertainty_spec.taps:
                if tap.name != "last":
                    num_out_channels_sum += tap.out_channels
            for tap in uncertainty_spec.taps:
                if tap.name != "last":

                    # implicitly assume that if_conditional = True when we strip extra channels
                    # and for sensible use of FNO, we should use if_conditional = True
                    num_retained_channels = self.num_physical_channels if tap.name == "after_last_decoder" else None
                    self.unc_heads[tap.name] = GAP2dAndFC(
                        in_channels=tap.in_channels,
                        out_channels=tap.out_channels,
                        num_layers=tap.num_layers,
                        hidden_channels=tap.hidden_channels,
                        make_channel_first_input=tap.make_channel_first_input,
                        use_end_activation=tap.use_end_activation,
                        time_as_second_dim=True, # can potentially be a config option later
                        num_retained_channels=num_retained_channels,
                        )
                    
                else:
                    self.unc_heads[tap.name] = FCs(
                                            in_channels=num_out_channels_sum,
                                            out_channels=tap.out_channels,
                                            num_layers=tap.num_layers,
                                            hidden_channels=tap.hidden_channels,
                                            use_end_activation=tap.use_end_activation # should generally be False
                                        )
                self.unc_taps_by_block.setdefault(tap.block_idx, []).append(tap)

    def _add_unc_outputs(self, unc_outputs, block_idx, x):
        if not self.uncertainty_enabled:
            return None

        taps = self.unc_taps_by_block.get(block_idx, [])
        for tap in taps:
            detach = self.uncertainty_spec.detach_features if tap.detach is None else tap.detach
            x_unc = x.detach() if detach else x
            if block_idx < 5: # input to block 5 is already unpadded
                x_unc = self._unpad(x_unc)

            out = self.unc_heads[tap.name](x_unc)
            unc_outputs.append(out)

    def _aggregate_unc_outputs(self, unc_outputs, block_idx):
        if len(unc_outputs) == 0:
            raise RuntimeError("return_uncertainty=True but no uncertainty taps" \
            " are enabled")

        stacked = torch.cat(unc_outputs, dim=-1)
        last_tap = self.unc_taps_by_block.get(block_idx, [])
        if len(last_tap) != 1:
            raise RuntimeError("Exactly one final uncertainty tap is required")
        last_tap = last_tap[0]
        detach = self.uncertainty_spec.detach_features if last_tap.detach is None else last_tap.detach
        stacked_unc = stacked.detach() if detach else stacked

        log_sigma = self.unc_heads[last_tap.name](stacked_unc)

        if self.uncertainty_spec.clamp_log_sigma:
            if self.uncertainty_spec.min_log_sigma is None and self.uncertainty_spec.max_log_sigma is None:
                raise ValueError("Invalid uncertainty spec. " \
                "Tried to clamp log sigma with both min_log_sigma " \
                "and max_log_sigma = None")
            log_sigma = torch.clamp(
                log_sigma,
                min=self.uncertainty_spec.min_log_sigma,
                max=self.uncertainty_spec.max_log_sigma,
            )

        return log_sigma

    def _linear_channel_first(self, layer, x):
        permute_to_last = [0] + list(range(2, x.dim())) + [1]
        permute_to_channel = [0, x.dim() - 1] + list(range(1, x.dim() - 1))
        return layer(x.permute(*permute_to_last)).permute(*permute_to_channel)

    def _pad(self, x):
        if self.padding == 0:
            return x
        return F.pad(x, [0, self.padding] * self.n_dims)

    def _unpad(self, x):
        if self.padding == 0:
            return x
        slices = [slice(None), slice(None)] + [slice(None, -self.padding)] * self.n_dims
        return x[tuple(slices)]

    def forward(self, x, grid, return_features=False, return_uncertainty=False):
        # x is [batch, spatial..., time, initial_step * channels].
        # grid is [batch, spatial..., time, spatial_dim + time].
        #print("input shape: ", x.shape)
        unc_outputs = []
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, x.dim() - 1, *range(1, x.dim() - 1))
        #print("shape after encoding and before padding: ", x.shape) # [B, W, X,..., T]
        x = self._pad(x)

        #print("shape after padding: ", x.shape) # [B, W, X + pad,..., T + pad]
        self._add_unc_outputs(unc_outputs, 0, x)

        features = []
        for idx, (conv, w) in enumerate(zip(self.convs, self.ws)):
            x = conv(x) + self._linear_channel_first(w, x)
            features.append(x)
            if idx < len(self.convs) - 1:
                x = F.gelu(x)
            self._add_unc_outputs(unc_outputs, idx + 1, x)
            #print(f"shape after Fourier block {idx}: ", x.shape) # [B, W, X + pad,..., T + pad]

        x = self._unpad(x)
        x = x.permute(0, *range(2, x.dim()), 1)
        #print("shape immediately before decoder layer 0: ", x.shape) # [B, X,..., T, W]
        x = self.fc1(x)
        x = F.gelu(x)
        self._add_unc_outputs(unc_outputs, 5, x)
        #print("shape after decoder layer 0: ", x.shape) # [B, X,..., T, 128]
        features.append(x)
        x = self.fc2(x)
        self._add_unc_outputs(unc_outputs, 6, x)
        #print("shape after decoder layer 1: ", x.shape) # [B, X,..., T, C + num_pde_param]
        if return_uncertainty:
            log_sigma = self._aggregate_unc_outputs(unc_outputs, block_idx=7)
            return x, log_sigma
        if return_features:
            return x, features[-1]
        return x


class FNO1dFullTrajectory(FNOFullTrajectory):
    def __init__(self, num_channels, modes1=16, modes2=16, width=64, 
                 initial_step=1, padding=2,
                 num_physical_channels: int | None = None,
                 uncertainty_spec: UncertaintySpec | None = None):
        super(FNO1dFullTrajectory, self).__init__(num_channels, 
                                                  2, 
                                                  (modes1, modes2), 
                                                  width, 
                                                  initial_step, 
                                                  padding,
                                                  num_physical_channels=num_physical_channels,
                                                  uncertainty_spec=uncertainty_spec)


class FNO2dFullTrajectory(FNOFullTrajectory):
    def __init__(self, num_channels, modes1=12, modes2=12, modes3=12, width=64, initial_step=1, padding=2,
                 num_physical_channels: int | None = None):
        super(FNO2dFullTrajectory, self).__init__(num_channels, 3, (modes1, modes2, modes3), width, initial_step, padding,
                                                  num_physical_channels=num_physical_channels)


class FNO3dFullTrajectory(FNOFullTrajectory):
    def __init__(self, num_channels, modes1=8, modes2=8, modes3=8, modes4=8, width=32, initial_step=1, padding=2,
                 num_physical_channels: int | None = None):
        super(FNO3dFullTrajectory, self).__init__(num_channels, 4, (modes1, modes2, modes3, modes4), width,
                                                  initial_step, padding,
                                                  num_physical_channels=num_physical_channels)
