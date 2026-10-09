import torch
import torch.nn as nn
import torch.nn.functional as F

from pquant.core.constants import QuantizationGranularity
from pquant.core.torch.fixed_point_quantizer import get_fixed_quantizer
from pquant.core.torch.hgq_quantizer import HGQQuantizer

ROW_WISE_DATA = (QuantizationGranularity.PER_TOKEN, QuantizationGranularity.PER_GROUP)


def _register_final_compression_flag(module):
    module.register_buffer("final_compression_done", torch.tensor(False))
    module._final_compression_done = False
    module.register_load_state_dict_post_hook(_sync_final_compression_flag)


def _sync_final_compression_flag(module, incompatible_keys=None):
    module._final_compression_done = bool(module.final_compression_done)


def _mark_final_compression_done(module):
    module.final_compression_done.fill_(True)
    module._final_compression_done = True


def quantizer_options(config):
    q = config.quantization_parameters
    return {
        "group_size": q.group_size,
        "dynamic_inference": q.dynamic_data_inference,
        "unclamped_bit_split": q.unclamped_bit_split,
    }


class Quantizer(nn.Module):
    # defaults for quantizers saved before these options existed
    group_size = 128
    dynamic_inference = False
    unclamped_bit_split = False
    shape = None

    def __init__(
        self,
        k,
        i,
        f,
        overflow,
        round_mode,
        is_heterogeneous,
        is_data=False,
        granularity=QuantizationGranularity.PER_TENSOR,
        hgq_gamma=0,
        place="datalane",
        dynamic_data=True,
        shape=None,
        group_size=128,
        dynamic_inference=False,
        unclamped_bit_split=False,
    ):
        super().__init__()

        self.overflow = overflow
        self.round_mode = round_mode
        self.use_hgq = is_heterogeneous
        self.is_data = is_data
        self.dynamic_data = dynamic_data
        self.dynamic_inference = dynamic_inference
        self.unclamped_bit_split = unclamped_bit_split
        self.group_size = int(group_size)
        self.granularity = QuantizationGranularity(granularity).value
        self.shape = None if shape is None else tuple(shape)

        # Params even when using HGQ, as they are used during hls4ml conversion
        param_shape = () if (is_data or self.use_hgq) else self.compute_weight_param_shape(shape)
        self.k = torch.nn.Parameter(torch.full(param_shape, float(k)), requires_grad=False)
        self.i = torch.nn.Parameter(torch.full(param_shape, float(i)), requires_grad=False)
        self.f = torch.nn.Parameter(torch.full(param_shape, float(f)), requires_grad=False)
        self.b = torch.nn.Parameter(torch.full(param_shape, float(i + k + f)), requires_grad=False)
        self.quantizer = create_quantizer(
            k,
            i,
            f,
            self.overflow,
            self.round_mode,
            self.use_hgq,
            self.is_data,
            granularity=self.granularity,
            gamma=hgq_gamma,
        )
        self.is_pretraining = True
        self.hgq_gamma = hgq_gamma
        _register_final_compression_flag(self)

    @property
    def grouped(self):
        if self.granularity != QuantizationGranularity.PER_GROUP or self.use_hgq:
            return False
        return self.is_data or (self.shape is not None and len(self.shape) == 2)

    def _group(self, x):
        """(..., n) -> (..., ceil(n / g), g), zero padded; returns the view and n."""
        n = x.shape[-1]
        pad = (-n) % self.group_size
        if pad:
            x = F.pad(x, (0, pad))
        return x.reshape(*x.shape[:-1], -1, self.group_size), n

    @staticmethod
    def _ungroup(xg, n):
        x = xg.reshape(*xg.shape[:-2], -1)
        return x if x.shape[-1] == n else x[..., :n].contiguous()

    def expand_bits(self, bits, shape):
        if self.grouped and bits.ndim == 3 and len(shape) == 2:
            bits = bits.expand(bits.shape[0], bits.shape[1], self.group_size).reshape(bits.shape[0], -1)
            return bits[:, : shape[1]]
        return bits

    def set_granularity(self, granularity, group_size=None):
        if self.use_hgq:
            raise ValueError("set_granularity is not supported for HGQ quantizers")
        if not self.is_data and self.shape is None:
            raise ValueError("set_granularity needs the weight shape, construct the Quantizer with a shape")
        self.granularity = QuantizationGranularity(granularity).value
        if group_size is not None:
            self.group_size = int(group_size)
        param_shape = () if self.is_data else self.compute_weight_param_shape(self.shape)
        values = {name: float(getattr(self, name).detach().flatten()[0]) for name in ("k", "i", "f")}
        device, dtype = self.k.device, self.k.dtype
        for name, value in values.items():
            setattr(
                self, name, nn.Parameter(torch.full(param_shape, value, device=device, dtype=dtype), requires_grad=False)
            )
        self.b = nn.Parameter(torch.full(param_shape, sum(values.values()), device=device, dtype=dtype), requires_grad=False)

    def get_quantization_bits(self):
        if self.use_hgq:
            return self.quantizer.k, self.quantizer.i, self.quantizer.f
        else:
            return self.k, self.i, self.f

    def get_total_bits(self, shape):
        if self.use_hgq:
            return self.quantizer.bits_(shape)
        else:
            b = self.expand_bits(self.i + self.f + self.k, shape)
            return torch.ones(shape).to(b.device) * b

    def _sync_hgq_mirror_bits(self):
        if not self.quantizer.built:
            return
        with torch.no_grad():
            k, i, f = self.quantizer.k.detach(), self.quantizer.i.detach(), self.quantizer.f.detach()
            self.k.data = k.clone()
            self.i.data = i.clone()
            self.f.data = f.clone()
            self.b.data = k + i + f

    def set_quantization_bits(self, i, f):
        if self.use_hgq:
            self.quantizer.set_bits(i, f)
        else:
            self.i.data = torch.as_tensor(i, dtype=self.i.dtype, device=self.i.device).broadcast_to(self.i.shape).clone()
            self.f.data = torch.as_tensor(f, dtype=self.f.dtype, device=self.f.device).broadcast_to(self.f.shape).clone()

    def post_pre_train_function(self):
        self.is_pretraining = False

    def calculate_bits_from_abs(self, abs_x):
        m = torch.ceil(torch.log2(abs_x + 1e-6))
        if self.unclamped_bit_split:
            return m, self.b - self.k.to(m.device) - m
        int_bits = torch.clamp(m, min=0).clamp(max=self.b - self.k.to(m.device))
        frac_bits = torch.clamp(self.b - int_bits - self.k, min=0)
        return int_bits, frac_bits

    @property
    def data_range_is_dynamic(self):
        return self.is_data and self.dynamic_data and (self.training or self.dynamic_inference)

    def compute_data_dynamic_bits(self, x):
        if not self.data_range_is_dynamic:
            _, i, f = self.get_quantization_bits()
            return i, f
        if self.granularity in ROW_WISE_DATA:
            abs_x = torch.amax(torch.abs(x), dim=-1, keepdim=True)
        else:
            abs_x = torch.amax(torch.abs(x))
        return self.calculate_bits_from_abs(abs_x)

    def compute_weight_param_shape(self, shape):
        if shape is None or self.granularity == QuantizationGranularity.PER_TENSOR or len(shape) == 1:
            return ()
        elif self.granularity == QuantizationGranularity.PER_TOKEN:
            return ()
        elif self.granularity == QuantizationGranularity.PER_GROUP and len(shape) == 2:
            return (shape[0], -(-shape[1] // self.group_size), 1)
        elif self.granularity in (QuantizationGranularity.PER_CHANNEL, QuantizationGranularity.PER_GROUP):
            return (shape[0],) + (1,) * (len(shape) - 1)  # Channels first
        else:
            return shape

    def compute_weight_dynamic_bits(self, x):
        per_tensor = self.granularity in (QuantizationGranularity.PER_TENSOR, QuantizationGranularity.PER_TOKEN)
        if per_tensor or x.ndim == 1 or not self.training:
            _, i, f = self.get_quantization_bits()
            return i, f
        if self.grouped:
            abs_x = torch.amax(torch.abs(x), dim=-1, keepdim=True)
        elif self.granularity in (QuantizationGranularity.PER_CHANNEL, QuantizationGranularity.PER_GROUP):
            if x.ndim == 2:
                abs_x = torch.amax(torch.abs(x), dim=1, keepdim=True)
            elif x.ndim == 3:
                abs_x = torch.amax(torch.abs(x), dim=(1, 2), keepdim=True)
            elif x.ndim == 4:
                abs_x = torch.amax(torch.abs(x), dim=(1, 2, 3), keepdim=True)
        elif self.granularity == QuantizationGranularity.PER_WEIGHT:
            abs_x = torch.abs(x)
        else:
            raise ValueError("The selected granularity is not supported.")
        return self.calculate_bits_from_abs(abs_x)

    def compute_dynamic_bits(self, x):
        if self.is_data:
            return self.compute_data_dynamic_bits(x)
        return self.compute_weight_dynamic_bits(x)

    def _remember_bits(self, i, f):
        with torch.no_grad():
            if self.is_data and self.granularity in ROW_WISE_DATA and i.shape != self.i.shape:
                i_max = i.detach().amax()
                self.i.copy_(i_max)
                self.f.copy_(self.b - self.k - i_max)
            elif self.i.shape == i.shape and self.f.shape == f.shape:
                self.i.copy_(i)
                self.f.copy_(f)
            else:
                self.i.data = i
                self.f.data = f

    def forward(self, x):
        if self.use_hgq:
            return self.quantizer(x, training=self.training)
        grouped = self.grouped and x.ndim > 1
        if grouped:
            x, n = self._group(x)
        if self._final_compression_done and not (self.data_range_is_dynamic and self.dynamic_inference):
            x = self.quantizer(x, k=self.k, i=self.i, f=self.f, training=False)
        else:
            i, f = self.compute_dynamic_bits(x)
            if self.training and not self._final_compression_done:
                self._remember_bits(i, f)
            x = self.quantizer(x, k=self.k, i=i, f=f, training=self.training and not self._final_compression_done)
        if grouped:
            x = self._ungroup(x, n)
        return x

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return 0.0
        return self.quantizer.regularization_loss()

    def post_epoch_function(self):
        if self.use_hgq and self.quantizer.built:
            self.quantizer.post_epoch_constraint_apply()

    def apply_final_compression(self):
        if self.use_hgq and not self.quantizer.built:
            return
        if self.use_hgq:
            with torch.no_grad():
                self.quantizer._f.data.clamp_(self.quantizer.f_min, self.quantizer.f_max)
                if self.quantizer.overflow_mode != "WRAP":
                    self.quantizer._i.data.clamp_(self.quantizer.i_min, self.quantizer.i_max)
            self._sync_hgq_mirror_bits()
            _mark_final_compression_done(self)
            return
        _, i, f = self.get_quantization_bits()
        self.i.data = i
        self.f.data = f
        self.b.data = i + f
        _mark_final_compression_done(self)


def create_quantizer(
    k, i, f, overflow, round_mode, is_heterogeneous, is_data, granularity=QuantizationGranularity.PER_WEIGHT, gamma=1e-8
):
    if is_heterogeneous:
        return HGQQuantizer(
            k0=k,
            i0=i,
            f0=f,
            overflow_mode=overflow,
            round_mode=round_mode,
            is_data=is_data,
            granularity=granularity,
            gamma=gamma,
            # hgq's defaults: data lanes let their WRAP integer bits decay 0.01 per step, weights track exactly
            i_decay_speed=0.01 if is_data else float("inf"),
        )
    else:
        return get_fixed_quantizer(round_mode=round_mode, overflow_mode=overflow)
