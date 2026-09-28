import os

import numpy as np
import pytest
import torch
from torch import nn

os.environ["KERAS_BACKEND"] = "torch"

from pquant.activations import PQActivation  # noqa: E402
from pquant.layers import (  # noqa: E402
    PQAvgPool1d,
    PQAvgPool2d,
    PQBatchNorm1d,
    PQBatchNorm2d,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQLayerNorm,
    PQMultiheadAttention,
    apply_final_compression,
    post_pretrain_functions,
    pre_finetune_functions,
)

from pquant import (  # noqa: E402
    ap_config,
    autosparse_config,
    cs_config,
    dst_config,
    mdmm_config,
    pdp_config,
    wanda_config,
)
from pquant.core.torch.layers import get_model_losses, post_epoch_functions  # noqa: E402
from pquant.core.torch.quantizer import Quantizer  # noqa: E402
from pquant.core.torch.train import train_model  # noqa: E402

BATCH_SIZE = 2
OUT_FEATURES = 8
IN_FEATURES = 4
KERNEL_SIZE = 3
STEPS = 6


class SingleLayerModel(nn.Module):
    def __init__(self, layer, is_mha=False):
        super().__init__()
        self.layer = layer
        self.is_mha = is_mha

    def forward(self, x):
        if self.is_mha:
            x, _ = self.layer(x, x, x)
        else:
            x = self.layer(x)
        return x


def build_model_and_input(layer_type, config):
    if layer_type == "dense":
        layer = PQDense(config, IN_FEATURES, OUT_FEATURES)
        x = torch.randn(BATCH_SIZE, IN_FEATURES)
    elif layer_type == "conv1d":
        layer = PQConv1d(config, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, padding=1)
        x = torch.randn(BATCH_SIZE, IN_FEATURES, STEPS)
    elif layer_type == "conv2d":
        layer = PQConv2d(config, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, padding=1)
        x = torch.randn(BATCH_SIZE, IN_FEATURES, STEPS, STEPS)
    elif layer_type == "batchnorm1d":
        layer = PQBatchNorm1d(config, IN_FEATURES)
        x = torch.randn(BATCH_SIZE, IN_FEATURES, STEPS)
    elif layer_type == "batchnorm2d":
        layer = PQBatchNorm2d(config, IN_FEATURES)
        x = torch.randn(BATCH_SIZE, IN_FEATURES, STEPS, STEPS)
    elif layer_type == "layernorm":
        layer = PQLayerNorm(config, IN_FEATURES)
        x = torch.randn(BATCH_SIZE, STEPS, IN_FEATURES)
    elif layer_type == "avgpool1d":
        layer = PQAvgPool1d(config, kernel_size=2)
        x = torch.randn(BATCH_SIZE, IN_FEATURES, STEPS)
    elif layer_type == "avgpool2d":
        layer = PQAvgPool2d(config, kernel_size=2)
        x = torch.randn(BATCH_SIZE, IN_FEATURES, STEPS, STEPS)
    elif layer_type.startswith("activation_"):
        layer = PQActivation(config, activation=layer_type.replace("activation_", ""), quantize_output=True)
        x = torch.randn(BATCH_SIZE, IN_FEATURES)
    elif layer_type == "mha":
        layer = PQMultiheadAttention(config, embed_dim=IN_FEATURES, num_heads=2)
        x = torch.randn(STEPS, BATCH_SIZE, IN_FEATURES)
        return SingleLayerModel(layer, is_mha=True), x
    else:
        raise ValueError(f"unknown layer kind {layer_type}")

    return SingleLayerModel(layer), x


STAGE_FLAGS = ("is_pretraining", "is_finetuning", "final_compression_done")


def randomize_state(model, seed):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, t in model.state_dict().items():
            if any(k in name for k in STAGE_FLAGS):
                continue
            if not torch.is_floating_point(t):
                continue
            t.copy_(torch.rand(t.shape, generator=gen, device="cpu") + 0.5)


def advance_to_stage(model, config, stage):
    if stage == "initial":
        return
    post_pretrain_functions(model, config)
    if stage == "post_pretrain":
        return
    pre_finetune_functions(model)
    if stage == "pre_finetune":
        return
    apply_final_compression(model)
    assert stage == "final_compression"


def make_config(use_hgq):
    config = dst_config()
    config.quantization_parameters.enable_quantization = True
    config.quantization_parameters.use_high_granularity_quantization = use_hgq
    return config


LAYER_TYPES = [
    "dense",
    "conv1d",
    "conv2d",
    "batchnorm1d",
    "batchnorm2d",
    "layernorm",
    "avgpool1d",
    "avgpool2d",
    "activation_relu",
    "activation_tanh",
    "activation_hard_tanh",
    "mha",
]
STAGES = ["initial", "post_pretrain", "pre_finetune", "final_compression"]


@pytest.mark.parametrize("use_hgq", [False, True], ids=["kif", "hgq"])
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("layer_type", LAYER_TYPES)
def test_state_dict_roundtrip(tmp_path, layer_type, stage, use_hgq):
    torch.manual_seed(0)
    config = make_config(use_hgq)

    model, x = build_model_and_input(layer_type, config)
    model(x)  # HGQ quantizers build lazily on first forward
    advance_to_stage(model, config, stage)
    randomize_state(model, seed=42)

    path = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), path)
    torch.manual_seed(1)
    fresh_config = make_config(use_hgq)
    fresh, _ = build_model_and_input(layer_type, fresh_config)
    fresh(x)
    advance_to_stage(fresh, fresh_config, stage)
    missing, unexpected = fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert not missing and not unexpected

    saved = model.state_dict()
    reloaded = fresh.state_dict()
    assert saved.keys() == reloaded.keys()
    for name in saved:
        np.testing.assert_array_equal(
            reloaded[name].cpu(), saved[name].cpu(), err_msg=f"state mismatch: {name}", strict=True
        )

    model.eval()
    fresh.eval()
    with torch.no_grad():
        np.testing.assert_array_equal(fresh(x).cpu(), model(x).cpu(), strict=True)


PRUNING_CONFIGS = {
    "dst": dst_config,
    "pdp": pdp_config,
    "wanda": wanda_config,
    "cs": cs_config,
    "autosparse": autosparse_config,
    "activation_pruning": ap_config,
    "mdmm": mdmm_config,
}


def make_pruning_config(method):
    config = PRUNING_CONFIGS[method]()
    config.quantization_parameters.enable_quantization = True
    if method in ("wanda", "activation_pruning"):
        # Reach the collection window within the few epochs these tests step through
        config.pruning_parameters.t_start_collecting_batch = 1
        config.pruning_parameters.t_delta = 2
    return config


def collect_one_window(model, x, config, epochs=3):
    """Run the model through one statistics-collection window and stop as soon as it closes.

    The last epoch skips ``post_epoch_functions`` on purpose. WANDA sets ``done`` once its mask is computed,
    but activation pruning only resets ``t`` and starts over, so one more post-epoch call would put it back
    into collection."""
    model.train()
    post_pretrain_functions(model, config)
    for epoch in range(epochs):
        for _ in range(3):
            model(x)
        if epoch < epochs - 1:
            post_epoch_functions(model, epoch, epochs)


@pytest.mark.parametrize("method", list(PRUNING_CONFIGS))
@pytest.mark.parametrize("stage", STAGES)
def test_state_dict_roundtrip_per_pruning_method(tmp_path, method, stage):
    torch.manual_seed(0)
    config = make_pruning_config(method)
    model, x = build_model_and_input("dense", config)
    model(x)
    advance_to_stage(model, config, stage)
    randomize_state(model, seed=42)

    path = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), path)

    torch.manual_seed(1)
    fresh_config = make_pruning_config(method)
    fresh, _ = build_model_and_input("dense", fresh_config)
    fresh(x)
    advance_to_stage(fresh, fresh_config, stage)
    missing, unexpected = fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert not missing and not unexpected

    saved = model.state_dict()
    reloaded = fresh.state_dict()
    assert saved.keys() == reloaded.keys()
    for name in saved:
        np.testing.assert_array_equal(
            reloaded[name].cpu(), saved[name].cpu(), err_msg=f"state mismatch: {name}", strict=True
        )

    model.eval()
    fresh.eval()
    with torch.no_grad():
        np.testing.assert_array_equal(fresh(x).cpu(), model(x).cpu(), strict=True)


def test_final_compression_mirror_follows_load(tmp_path):
    """The forward pass reads ``final_compression_done`` from a Python bool that mirrors the checkpointed buffer.
    Loading a compressed checkpoint into an uncompressed model must update that bool as well. Otherwise the
    quantizers would quantize the already-compressed weights a second time."""
    torch.manual_seed(0)
    config = make_config(use_hgq=False)
    model, x = build_model_and_input("dense", config)
    model(x)
    advance_to_stage(model, config, "final_compression")
    path = tmp_path / "compressed.pt"
    torch.save(model.state_dict(), path)

    torch.manual_seed(0)
    fresh_config = make_config(use_hgq=False)
    fresh, _ = build_model_and_input("dense", fresh_config)
    fresh(x)
    advance_to_stage(fresh, fresh_config, "pre_finetune")  # deliberately NOT compressed
    quantizers = [m for m in fresh.modules() if isinstance(m, Quantizer)]
    assert quantizers and not any(q._final_compression_done for q in quantizers)

    fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)

    for q in quantizers:
        assert bool(q.final_compression_done) is True
        assert q._final_compression_done is True, "Python mirror did not follow the checkpointed buffer"

    model.eval()
    fresh.eval()
    with torch.no_grad():
        np.testing.assert_array_equal(fresh(x).cpu(), model(x).cpu(), strict=True)


@pytest.mark.parametrize("method", ["wanda", "activation_pruning"])
def test_collecting_mirror_follows_load(tmp_path, method):
    """``collecting`` is a Python bool computed from the checkpointed ``t`` and ``done`` buffers.
    Loading a checkpoint saved after its collection window into a model that is still inside one must
    recompute that bool. Otherwise the layer keeps collecting and overwrites the restored mask."""
    torch.manual_seed(0)
    config = make_pruning_config(method)
    model, x = build_model_and_input("dense", config)
    model(x)
    collect_one_window(model, x, config)
    pruning_layer = model.layer.pruning_layer
    assert not pruning_layer.collecting, "saved model should have finished its collection window"
    path = tmp_path / "collected.pt"
    torch.save(model.state_dict(), path)

    torch.manual_seed(1)
    fresh_config = make_pruning_config(method)
    fresh, _ = build_model_and_input("dense", fresh_config)
    fresh(x)
    fresh.train()
    post_pretrain_functions(fresh, fresh_config)
    post_epoch_functions(fresh, 0, 4)  # advances t past t_start_collecting_batch
    fresh_layer = fresh.layer.pruning_layer
    assert fresh_layer.collecting, "fresh model should be mid-collection before the load"

    fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert not fresh_layer.collecting, "Python mirror did not follow the checkpointed counters"

    # With collection off, further forwards must not touch any of the restored counters or masks
    before = {k: v.clone() for k, v in fresh_layer.state_dict().items()}
    fresh.train()
    for _ in range(3):
        fresh(x)
    for name, value in fresh_layer.state_dict().items():
        np.testing.assert_array_equal(
            value.cpu(), before[name].cpu(), err_msg=f"{name} was overwritten by further collection"
        )


def _fixed_batches(x, n=2):
    torch.manual_seed(7)
    return [(x.clone(), torch.randn(x.shape[0], OUT_FEATURES)) for _ in range(n)]


def _train_one_epoch(model, batches=None, optimizer=None, **kwargs):
    loss_function = nn.MSELoss()
    for xb, yb in batches:
        optimizer.zero_grad()
        loss = loss_function(model(xb), yb) + get_model_losses(model, torch.zeros(()))
        loss.backward()
        optimizer.step()


def _noop_validate(model, **kwargs):
    pass


def _run_training(model, config, batches, pretraining, epochs, finetuning, rounds=None):
    config.training_parameters.pretraining_epochs = pretraining
    config.training_parameters.epochs = epochs
    config.training_parameters.fine_tuning_epochs = finetuning
    if rounds is not None:
        config.training_parameters.rounds = rounds
    # Plain SGD carries no optimizer state, so a resumed run can match an uninterrupted one exactly
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return train_model(
        model=model,
        config=config,
        train_func=_train_one_epoch,
        valid_func=_noop_validate,
        batches=batches,
        optimizer=optimizer,
        input_shape=(IN_FEATURES,),
    )


@pytest.mark.parametrize("method", list(PRUNING_CONFIGS))
def test_resume_from_checkpoint_matches_uninterrupted_training(tmp_path, method):
    """Resuming is just ``train_model`` with the remaining epochs. It always calls ``post_pretrain_functions``
    and ``pre_finetune_functions``, so a model loaded from a checkpoint ends up in the right stage by itself.
    Training for 1 pretraining and 2 pruning epochs, saving, then fine-tuning for 1 epoch on a freshly loaded
    model must give exactly the same result as training for 1 + 2 + 1 epochs in one go."""
    torch.manual_seed(0)
    reference, x = build_model_and_input("dense", make_pruning_config(method))
    reference(x)
    batches = _fixed_batches(x)
    _run_training(reference, make_pruning_config(method), batches, pretraining=1, epochs=2, finetuning=1)

    torch.manual_seed(0)
    interrupted, _ = build_model_and_input("dense", make_pruning_config(method))
    interrupted(x)
    _run_training(interrupted, make_pruning_config(method), batches, pretraining=1, epochs=2, finetuning=0)
    path = tmp_path / "resume.pt"
    torch.save(interrupted.state_dict(), path)

    torch.manual_seed(123)  # different init: everything must come from the checkpoint
    resumed, _ = build_model_and_input("dense", make_pruning_config(method))
    resumed(x)
    resumed.load_state_dict(torch.load(path, weights_only=True), strict=True)
    # Only the epoch counts are set to 0; ``rounds`` is left as configured. ``train_model`` skips the post-round
    # functions when a round has no epochs, so they cannot run again on the resumed model.
    _run_training(resumed, make_pruning_config(method), batches, pretraining=0, epochs=0, finetuning=1)

    pruning_layer = resumed.layer.pruning_layer
    assert pruning_layer.is_finetuning, "train_model should have put the resumed model into the finetuning stage"
    assert not pruning_layer.is_pretraining

    expected = reference.state_dict()
    actual = resumed.state_dict()
    assert expected.keys() == actual.keys()
    for name in expected:
        np.testing.assert_allclose(
            actual[name].cpu().numpy(),
            expected[name].cpu().numpy(),
            rtol=0,
            atol=0,
            err_msg=f"resumed training diverged from uninterrupted training: {name}",
        )

    reference.eval()
    resumed.eval()
    with torch.no_grad():
        np.testing.assert_array_equal(resumed(x).cpu(), reference(x).cpu(), strict=True)
