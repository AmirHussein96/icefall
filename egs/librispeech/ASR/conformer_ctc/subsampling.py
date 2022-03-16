# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-1)//2 - 1)//2, which approximates T' == T//4

    It is based on
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py  # noqa
    """

    def __init__(self, idim: int, odim: int) -> None:
        """
        Args:
          idim:
            Input dim. The input shape is (N, T, idim).
            Caution: It requires: T >=7, idim >=7
          odim:
            Output dim. The output shape is (N, ((T-1)//2 - 1)//2, odim)
        """
        assert idim >= 7
        super().__init__()
        self.conv = nn.Sequential(
            ScaledConv2d(
                in_channels=1, out_channels=odim, kernel_size=3, stride=2
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
            ScaledConv2d(
                in_channels=odim, out_channels=odim, kernel_size=3, stride=2
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
        )
        self.out = ScaledLinear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        # set learn_eps=False because out_norm is preceded by `out`, and `out`
        # itself has learned scale, so the extra degree of freedom is not
        # needed.
        self.out_norm = BasicNorm(odim, learn_eps=False)
        # constrain median of output to be close to zero.
        self.out_balancer = ActivationBalancer(channel_dim=-1,
                                               min_positive=0.45,
                                               max_positive=0.55)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, ((T-1)//2 - 1)//2, odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)
        # Now x is of shape (N, odim, ((T-1)//2 - 1)//2, ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        # Now x is of shape (N, ((T-1)//2 - 1))//2, odim)
        x = self.out_norm(x)
        x = self.out_balancer(x)
        return x


class VggSubsampling(nn.Module):
    """Trying to follow the setup described in the following paper:
    https://arxiv.org/pdf/1910.09799.pdf

    This paper is not 100% explicit so I am guessing to some extent,
    and trying to compare with other VGG implementations.

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-1)//2 - 1)//2, which approximates T' = T//4
    """

    def __init__(self, idim: int, odim: int) -> None:
        """Construct a VggSubsampling object.

        This uses 2 VGG blocks with 2 Conv2d layers each,
        subsampling its input by a factor of 4 in the time dimensions.

        Args:
          idim:
            Input dim. The input shape is (N, T, idim).
            Caution: It requires: T >=7, idim >=7
          odim:
            Output dim. The output shape is (N, ((T-1)//2 - 1)//2, odim)
        """
        super().__init__()

        cur_channels = 1
        layers = []
        block_dims = [32, 64]

        # The decision to use padding=1 for the 1st convolution, then padding=0
        # for the 2nd and for the max-pooling, and ceil_mode=True, was driven by
        # a back-compatibility concern so that the number of frames at the
        # output would be equal to:
        #  (((T-1)//2)-1)//2.
        # We can consider changing this by using padding=1 on the
        # 2nd convolution, so the num-frames at the output would be T//4.
        for block_dim in block_dims:
            layers.append(
                torch.nn.Conv2d(
                    in_channels=cur_channels,
                    out_channels=block_dim,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                )
            )
            layers.append(torch.nn.ReLU())
            layers.append(
                torch.nn.Conv2d(
                    in_channels=block_dim,
                    out_channels=block_dim,
                    kernel_size=3,
                    padding=0,
                    stride=1,
                )
            )
            layers.append(
                torch.nn.MaxPool2d(
                    kernel_size=2, stride=2, padding=0, ceil_mode=True
                )
            )
            cur_channels = block_dim

        self.layers = nn.Sequential(*layers)

        self.out = nn.Linear(
            block_dims[-1] * (((idim - 1) // 2 - 1) // 2), odim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, ((T-1)//2 - 1)//2, odim)
        """
        x = x.unsqueeze(1)
        x = self.layers(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x





class ActivationBalancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor,
                channel_dim: int,
                min_positive: float, # e.g. 0.05
                max_positive: float, # e.g. 0.95
                max_factor: float, # e.g. 0.01
                min_abs: float, # e.g. 0.2
                max_abs: float, # e.g. 100.0
    ) -> Tensor:
        if x.requires_grad:
            if channel_dim < 0:
                channel_dim += x.ndim
            sum_dims = [d for d in range(x.ndim) if d != channel_dim]
            xgt0 = x > 0
            proportion_positive = torch.mean(xgt0.to(x.dtype), dim=sum_dims, keepdim=True)
            factor1 = ((min_positive - proportion_positive).relu() * (max_factor / min_positive)
                       if min_positive != 0.0 else 0.0)
            factor2 = ((proportion_positive - max_positive).relu() * (max_factor / (max_positive - 1.0))
                       if max_positive != 1.0 else 0.0)
            factor = factor1 + factor2
            if isinstance(factor, float):
                factor = torch.zeros_like(proportion_positive)

            mean_abs = torch.mean(x.abs(), dim=sum_dims, keepdim=True)
            below_threshold = (mean_abs < min_abs)
            above_threshold = (mean_abs > max_abs)

            ctx.save_for_backward(factor, xgt0, below_threshold, above_threshold)
            ctx.max_factor = max_factor
            ctx.sum_dims = sum_dims
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor) -> Tuple[Tensor, None, None, None, None, None, None]:
        factor, xgt0, below_threshold, above_threshold = ctx.saved_tensors
        dtype = x_grad.dtype
        scale_factor = ((below_threshold.to(dtype) - above_threshold.to(dtype)) *
                        (xgt0.to(dtype) - 0.5) * (ctx.max_factor * 2.0))

        neg_delta_grad = x_grad.abs() * (factor + scale_factor)
        return x_grad - neg_delta_grad, None, None, None, None, None, None


class BasicNorm(torch.nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm.  The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.

    So the idea is to introduce this large constant value as an explicit
    parameter, that takes the role of the "eps" in LayerNorm, so the network
    doesn't have to do this trick.  We make the "eps" learnable.

    Args:
       num_channels: the number of channels, e.g. 512.
      channel_dim: the axis/dimension corresponding to the channel,
        interprted as an offset from the input's ndim if negative.
        shis is NOT the num_channels; it should typically be one of
        {-2, -1, 0, 1, 2, 3}.
       eps: the initial "epsilon" that we add as ballast in:
             scale = ((input_vec**2).mean() + epsilon)**-0.5
          Note: our epsilon is actually large, but we keep the name
          to indicate the connection with conventional LayerNorm.
       learn_eps: if true, we learn epsilon; if false, we keep it
         at the initial value.
      eps_speed: a constant that determines how fast "eps" learns;
          with Adam and variants, this should probably be >= 1,
          e.g. 5.0.  For SGD and variants, probably a value less than one,
          like 0.1, would be suitable, to prevent instability.
    """
    def __init__(self,
                 num_channels: int,
                 channel_dim: int = -1,  # CAUTION: see documentation.
                 eps: float = 0.25,
                 learn_eps: bool = True,
                 eps_speed: float = 5.0):
        super(BasicNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.eps_speed = eps_speed
        if learn_eps:
            self.eps = nn.Parameter((torch.tensor(eps).log() / eps_speed).detach())
        else:
            self.register_buffer('eps', (torch.tensor(eps).log() / eps_speed).detach())


    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels
        scales = (torch.mean(x**2, dim=self.channel_dim, keepdim=True) +
                  (self.eps * self.eps_speed).exp()) ** -0.5
        return x * scales




class ScaledLinear(nn.Linear):
    """
    A modified version of nn.Linear where the parameters are scaled before
    use, via:
         weight = self.weight * (self.weight_scale * self.scale_speed).exp()
         bias = self.bias * (self.bias_scale * self.scale_speed).exp()

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        scale_speed: a factor that affects how fast the weight_scale
           and bias_scale learn; this value is suitable for Adam-type
           optimizers.
        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.

       Note: it uses the default initialization for the weight and bias,
       inherited from nn.Linear.  For modules with small fan-in, this
       may be larger than optimal.
    """
    def __init__(self, *args,
                 scale_speed: float = 5.0,
                 initial_scale: float = 1.0,
                 **kwargs):
        super(ScaledLinear, self).__init__(*args, **kwargs)
        initial_scale = (torch.tensor(initial_scale).log() / scale_speed)
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        self.scale_speed = scale_speed
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter('bias_scale', None)

        self._reset_parameters()  # Overrides the reset_parameters in nn.Linear

    def _reset_parameters(self):
        std = 0.05
        a = (3 ** 0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in ** -0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += (torch.tensor(scale / std).log() / self.scale_speed)

    def get_weight(self):
        return self.weight * (self.weight_scale * self.scale_speed).exp()

    def get_bias(self):
        return (None if self.bias is None else
                self.bias * (self.bias_scale * self.scale_speed).exp())

    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.linear(input, self.get_weight(),
                                          self.get_bias())


class ScaledConv1d(nn.Conv1d):
    def __init__(self, *args, scale_speed = 5.0,
                 initial_scale=1.0, **kwargs):
        super(ScaledConv1d, self).__init__(*args, **kwargs)
        self.scale_speed = scale_speed
        initial_scale = (torch.tensor(initial_scale).log() / scale_speed)
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter('bias_scale', None)
        self._reset_parameters()  # Overrides the reset_parameters in base class

    def _reset_parameters(self):
        std = 0.05
        a = (3 ** 0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in ** -0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += (torch.tensor(scale / std).log() / self.scale_speed)


    def get_weight(self):
        return self.weight * (self.weight_scale * self.scale_speed).exp()

    def get_bias(self):
        return (None if self.bias is None else
                self.bias * (self.bias_scale * self.scale_speed).exp())

    def forward(self, input: Tensor) -> Tensor:
        F = torch.nn.functional
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self.get_weight(), self.get_bias(), self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, self.get_weight(), self.get_bias(), self.stride,
                        self.padding, self.dilation, self.groups)



class ScaledConv2d(nn.Conv2d):
    def __init__(self, *args, scale_speed=5.0, initial_scale=1.0, **kwargs):
        super(ScaledConv2d, self).__init__(*args, **kwargs)
        self.scale_speed = scale_speed
        initial_scale = (torch.tensor(initial_scale).log() / scale_speed)
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter('bias_scale', None)
        self._reset_parameters()  # Overrides the reset_parameters in base class

    def _reset_parameters(self):
        std = 0.05
        a = (3 ** 0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in ** -0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += (torch.tensor(scale / std).log() / self.scale_speed)


    def get_weight(self):
        return self.weight * (self.weight_scale * self.scale_speed).exp()

    def get_bias(self):
        return (None if self.bias is None else
                self.bias * (self.bias_scale * self.scale_speed).exp())

    def _conv_forward(self, input, weight):
        F = torch.nn.functional
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.get_bias(), self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.get_bias(), self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.get_weight())




class ActivationBalancer(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to encourage, for
    each channel, that it is positive at least a proportion `threshold` of the
    time.  It does this by multiplying negative derivative values by up to
    (1+max_factor), and positive derivative values by up to (1-max_factor),
    interpolated from 1 at the threshold to those extremal values when none
    of the inputs are positive.


    Args:
           channel_dim: the dimension/axis corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           min_positive: the minimum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_positive: the maximum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_factor: the maximum factor by which we modify the derivatives for
              either the sign constraint or the magnitude constraint;
              e.g. with max_factor=0.02, the the derivatives would be multiplied by
              values in the range [0.98..1.02].
           min_abs:  the minimum average-absolute-value per channel, which
              we allow, before we start to modify the derivatives to prevent
              this.
           max_abs:  the maximum average-absolute-value per channel, which
               we allow, before we start to modify the derivatives to prevent
               this.
    """
    def __init__(self, channel_dim: int,
                 min_positive: float = 0.05,
                 max_positive: float = 0.95,
                 max_factor: float = 0.01,
                 min_abs: float = 0.2,
                 max_abs: float = 100.0):
        super(ActivationBalancer, self).__init__()
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.max_factor = max_factor
        self.min_abs = min_abs
        self.max_abs = max_abs

    def forward(self, x: Tensor) -> Tensor:
        return ActivationBalancerFunction.apply(x, self.channel_dim,
                                                self.min_positive, self.max_positive,
                                                self.max_factor, self.min_abs,
                                                self.max_abs)


def _double_swish(x: Tensor) -> Tensor:
    # double-swish, implemented/approximated as offset-swish
    return x * torch.sigmoid(x - 1.0)

class DoubleSwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x.detach())
        return _double_swish(x)

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        # TODO: can make this more efficient.
        x, = ctx.saved_tensors
        x.requires_grad = True
        with torch.enable_grad():
            y = _double_swish(x)
            y.backward(gradient=y_grad)
            return x.grad

class DoubleSwish(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return double-swish activation function which is an approximation to Swish(Swish(x)),
           that we approximate closely with x * sigmoid(x-1).
        """
        return DoubleSwishFunction.apply(x)



def _test_deriv_balancer_sign():
    channel_dim = 0
    probs = torch.arange(0, 1, 0.01)
    N = 1000
    x = 1.0 * (torch.rand(probs.numel(), N) < probs.unsqueeze(-1))
    x = x.detach()
    x.requires_grad = True
    m = ActivationBalancer(channel_dim=0, min_positive=0.05, max_positive=0.95,
                           max_factor=0.2, min_abs=0.0)

    y_grad = torch.sign(torch.randn(probs.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_deriv_balancer_sign: x = ", x)
    print("_test_deriv_balancer_sign: y grad = ", y_grad)
    print("_test_deriv_balancer_sign: x grad = ", x.grad)

def _test_deriv_balancer_magnitude():
    channel_dim = 0
    magnitudes = torch.arange(0, 1, 0.01)
    N = 1000
    x = torch.sign(torch.randn(magnitudes.numel(), N))  * magnitudes.unsqueeze(-1)
    x = x.detach()
    x.requires_grad = True
    m = ActivationBalancer(channel_dim=0,
                           min_positive=0.0, max_positive=1.0,
                           max_factor=0.2,
                           min_abs=0.2, max_abs=0.8)

    y_grad = torch.sign(torch.randn(magnitudes.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_deriv_balancer_magnitude: x = ", x)
    print("_test_deriv_balancer_magnitude: y grad = ", y_grad)
    print("_test_deriv_balancer_magnitude: x grad = ", x.grad)


def _test_basic_norm():
    num_channels = 128
    m = BasicNorm(num_channels=num_channels, channel_dim=1)

    x = torch.randn(500, num_channels)

    y = m(x)

    assert y.shape == x.shape
    x_rms = (x**2).mean().sqrt()
    y_rms = (y**2).mean().sqrt()
    print("x rms = ", x_rms)
    print("y rms = ", y_rms)
    assert y_rms < x_rms
    assert y_rms > 0.5 * x_rms





if __name__ == '__main__':
    _test_deriv_balancer_sign()
    _test_deriv_balancer_magnitude()
    _test_basic_norm()
