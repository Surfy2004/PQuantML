import torch
import torch.nn as nn


def _refresh_collecting_after_load(module, incompatible_keys=None):
    module._refresh_collecting()


class ActivationPruning(nn.Module):
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
        self.threshold = float(config.pruning_parameters.threshold)
        self.t_start_collecting_batch = int(self.config.pruning_parameters.t_start_collecting_batch)
        self.built = False
        self._collecting = False

    def build(self, input_shape):
        if self.built:
            return
        if self.layer_type in ("conv", "depthwise_conv"):
            if len(input_shape) == 3:
                shape = (input_shape[0], 1, 1)
            else:
                shape = (input_shape[0], 1, 1, 1)
        else:
            shape = (input_shape[0], 1)
        self.shape = shape
        n_channels = input_shape[0]
        self.register_buffer("mask", torch.ones(shape))
        self.register_buffer("mask_placeholder", torch.ones(shape))
        self.register_buffer("activations", torch.zeros(n_channels))
        self.register_buffer("batches_collected", torch.zeros((), dtype=torch.int32))
        self.register_buffer("t", torch.zeros((), dtype=torch.int32))
        self.register_load_state_dict_post_hook(_refresh_collecting_after_load)
        self.built = True
        self._refresh_collecting()

    @property
    def collecting(self):
        """True while output statistics are being gathered.

        This is a plain Python bool rather than a check on the counter buffers, so the forward pass does not
        sync with the device every step or break the torch.compile graph. It is recomputed only when the
        pruning stage changes or ``t`` reaches the start epoch.
        """
        return self._collecting

    def _refresh_collecting(self):
        self._collecting = (
            self.built
            and not self._is_pretraining
            and not self._is_finetuning
            and int(self.t.item()) >= self.t_start_collecting_batch
        )

    @torch.compiler.disable
    @torch.no_grad()
    def collect_output(self, output, training):
        # Kept out of torch.compile: the counter checks below branch on device scalars, which would break the
        # graph. Running eagerly only costs something during the collection window, not the rest of training.
        if not training or not self._collecting:
            return

        t_delta = int(self.config.pruning_parameters.t_delta)

        gt_zero = (output > 0).to(output.dtype)
        if self.layer_type == "linear":
            per_channel = gt_zero.mean(dim=0)
        else:
            axes = (0,) + tuple(range(2, output.dim()))
            per_channel = gt_zero.mean(dim=axes)

        self.activations.add_(per_channel.to(self.activations.dtype))
        self.batches_collected.add_(1)

        if int(self.batches_collected.item()) % t_delta == 0:
            denom = max(int(self.batches_collected.item()), 1)
            pct_active = self.activations / denom
            new_mask = (pct_active > self.threshold).to(self.mask_placeholder.dtype).reshape(self.shape)
            self.mask_placeholder.copy_(new_mask)
            self.activations.zero_()
            self.batches_collected.zero_()
            self.t.zero_()
            self._refresh_collecting()

    def forward(self, weight):
        if self._is_pretraining:
            return weight
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
        self.mask.copy_(self.mask_placeholder)
        self._refresh_collecting()
