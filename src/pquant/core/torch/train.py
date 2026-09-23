import torch

from pquant.core.torch.layers import (
    call_post_round_functions,
    post_epoch_functions,
    post_pretrain_functions,
    pre_epoch_functions,
    pre_finetune_functions,
    save_weights_functions,
)


def train_model(model, config, train_func, valid_func, input_shape=None, **kwargs):
    """
    Generic training loop, user provides training and validation functions.

    ``model`` may be wrapped with ``torch.compile``. ``train_func`` and ``valid_func`` receive the model as given.
    The calls made between them (``pre_epoch_functions``, ``post_epoch_functions``, ``post_pretrain_functions``
    and the other helpers imported from ``layers``) run on the underlying eager module, since they only walk the
    PQ layers and update their pruning and quantization state. Returns ``model`` as it was passed in.
    """
    epoch = torch.tensor(0)  # Keeps track of all the epochs completed
    training_config = config.training_parameters
    eager_model = getattr(model, "_orig_mod", model)
    if training_config.pretraining_epochs > 0:
        for e in range(training_config.pretraining_epochs):
            model.train()
            pre_epoch_functions(eager_model, e, training_config.pretraining_epochs)
            train_func(model, epoch=epoch, **kwargs)
            model.eval()
            valid_func(model, epoch=epoch, **kwargs)
            post_epoch_functions(eager_model, e, training_config.pretraining_epochs)
            epoch += 1
    post_pretrain_functions(
        eager_model, config, kwargs.get("trainloader", None), kwargs.get("loss_function", None), input_shape=input_shape
    )
    for r in range(training_config.rounds):
        for e in range(training_config.epochs):
            model.train()
            if r == 0 and training_config.save_weights_epoch == e:
                save_weights_functions(eager_model)
            pre_epoch_functions(eager_model, e, training_config.epochs)
            train_func(model, epoch=epoch, **kwargs)
            model.eval()
            valid_func(model, epoch=epoch, **kwargs)
            post_epoch_functions(eager_model, e, training_config.epochs)
            epoch += 1
        if training_config.epochs > 0:
            call_post_round_functions(eager_model, training_config.rewind, training_config.rounds, r)
    pre_finetune_functions(eager_model)
    if training_config.fine_tuning_epochs > 0:
        for e in range(training_config.fine_tuning_epochs):
            model.train()
            pre_epoch_functions(eager_model, e, training_config.fine_tuning_epochs)
            train_func(model, epoch=epoch, **kwargs)
            model.eval()
            valid_func(model, epoch=epoch, **kwargs)
            post_epoch_functions(eager_model, e, training_config.fine_tuning_epochs)
            epoch += 1
    return model
