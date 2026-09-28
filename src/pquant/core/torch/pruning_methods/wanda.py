import torch
import torch.nn as nn


def _refresh_collecting_after_load(module, incompatible_keys=None):
    module._refresh_collecting()


class Wanda(nn.Module):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__()
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.config = config
        self.act_type = "relu"
        self.layer_type = layer_type
        self._is_pretraining = True
        self._is_finetuning = False
        self.is_pretraining = True
        self.is_finetuning = False
        self._sparsity = float(self.config.pruning_parameters.sparsity)
        self.N = self.config.pruning_parameters.N
        self.M = self.config.pruning_parameters.M
        self.t_start_collecting_batch = int(self.config.pruning_parameters.t_start_collecting_batch)
        self.built = False
        self._collecting = False

    # Expose sparsity as a tensor so it supports torch tensor API (.cpu(), etc.)
    # while keeping the Python float for cheap internal arithmetic.
    @property
    def sparsity(self):
        return torch.tensor(self._sparsity)

    @sparsity.setter
    def sparsity(self, v):
        self._sparsity = float(v.detach().item()) if torch.is_tensor(v) else float(v)

    def build(self, input_shape):
        if self.built:
            return
        n_in = input_shape[0] if self.layer_type == "depthwise_conv" else input_shape[1]
        self.register_buffer("mask", torch.ones(tuple(input_shape)))
        self.register_buffer("inputs_sq_sum", torch.zeros(n_in))
        self.register_buffer("batches_collected", torch.zeros((), dtype=torch.int32))
        self.register_buffer("t", torch.zeros((), dtype=torch.int32))
        self.register_buffer("done", torch.zeros((), dtype=torch.bool))
        self.register_load_state_dict_post_hook(_refresh_collecting_after_load)
        self.built = True
        self._refresh_collecting()

    @property
    def collecting(self):
        """True while input statistics are being gathered.

        This is a plain Python bool rather than a check on the counter buffers, so the forward pass does not
        sync with the device every step or break the torch.compile graph. It is recomputed only when the
        pruning stage changes, ``t`` reaches the start epoch, or the mask is set.
        """
        return self._collecting

    def _refresh_collecting(self):
        self._collecting = (
            self.built
            and not self._is_pretraining
            and not self._is_finetuning
            and not bool(self.done.item())
            and int(self.t.item()) >= self.t_start_collecting_batch
        )

    @torch.compiler.disable
    @torch.no_grad()
    def collect_input(self, x, weight, training):
        # Kept out of torch.compile: the counter checks below branch on device scalars, which would break the
        # graph. Running eagerly only costs something during the collection window, not the rest of training.
        if not training or not self._collecting:
            return

        t_delta = int(self.config.pruning_parameters.t_delta)

        if self.layer_type == "linear":
            per_batch_sq = (x * x).sum(dim=0)
        else:
            axes = (0,) + tuple(range(2, x.dim()))
            per_batch_sq = (x * x).sum(dim=axes)

        self.inputs_sq_sum.add_(per_batch_sq.to(self.inputs_sq_sum.dtype))
        self.batches_collected.add_(1)

        if int(self.batches_collected.item()) == t_delta:
            norm = torch.sqrt(self.inputs_sq_sum)
            new_mask = self._compute_prune_mask(norm, weight)
            self.mask.copy_(new_mask.to(self.mask.dtype))
            self.done.fill_(True)
            self.inputs_sq_sum.zero_()
            self.batches_collected.zero_()
            self._collecting = False

    def _compute_prune_mask(self, norm, weight):
        if self.layer_type == "linear":
            return self._handle_linear(norm, weight)
        if self.layer_type == "depthwise_conv":
            return self._handle_depthwise_conv(norm, weight)
        return self._handle_conv(norm, weight)

    def _handle_linear(self, norm, weight):
        metric = weight.abs() * norm
        if self.N is not None and self.M is not None:
            metric_reshaped = metric.reshape(-1, self.M)
            weight_reshaped = weight.reshape(-1, self.M)
            mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self.N / self.M)
            return mask.reshape(weight.shape)
        metric_reshaped = metric.reshape(1, -1)
        weight_reshaped = weight.reshape(1, -1)
        mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self._sparsity)
        return mask.reshape(weight.shape)

    def _handle_conv(self, norm, weight):
        if weight.dim() == 3:
            norm_reshaped = norm.reshape(1, norm.shape[0], 1)
        else:
            norm_reshaped = norm.reshape(1, norm.shape[0], 1, 1)
        metric = weight.abs() * norm_reshaped
        if self.N is not None and self.M is not None:
            metric_reshaped = metric.reshape(-1, self.M)
            weight_reshaped = weight.reshape(-1, self.M)
            mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self.N / self.M)
            return mask.reshape(weight.shape)
        metric_reshaped = metric.reshape(metric.shape[0], -1)
        weight_reshaped = weight.reshape(weight.shape[0], -1)
        mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self._sparsity)
        return mask.reshape(weight.shape)

    def _handle_depthwise_conv(self, norm, weight):
        norm_reshaped = norm.reshape(norm.shape[0], 1, 1, 1)
        metric = weight.abs() * norm_reshaped
        metric_reshaped = metric.reshape(metric.shape[0], -1)
        weight_reshaped = weight.reshape(weight.shape[0], -1)
        mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self._sparsity)
        return mask.reshape(weight.shape)

    def get_mask(self, weight, metric, sparsity):
        d0, d1 = metric.shape
        keep_idxs = (
            torch.argsort(metric, dim=1, stable=True)[:, int(d1 * sparsity) :]
            + torch.arange(d0, device=metric.device)[:, None] * d1
        )
        keep_idxs = keep_idxs.flatten()
        kept_values = torch.zeros(weight.numel(), dtype=weight.dtype, device=weight.device)
        kept_values[keep_idxs] = weight.flatten()[keep_idxs]
        kept_values = kept_values.reshape(weight.shape)
        return (kept_values != 0).to(weight.dtype)

    def forward(self, weight):
        return self.mask.to(weight.dtype) * weight

    def get_hard_mask(self, weight=None):
        return self.mask

    def post_pre_train_function(self):
        self._is_pretraining = False
        self.is_pretraining = False
        self._refresh_collecting()

    def pre_epoch_function(self, epoch, total_epochs, **kwargs):
        pass

    def post_round_function(self):
        pass

    def pre_finetune_function(self):
        self._is_finetuning = True
        self.is_finetuning = True
        self._refresh_collecting()

    def calculate_additional_loss(self):
        return 0.0

    def get_layer_sparsity(self, weight):
        pass

    @torch.no_grad()
    def post_epoch_function(self, epoch, total_epochs, **kwargs):
        if not self._is_pretraining:
            self.t.add_(1)
        self._refresh_collecting()
