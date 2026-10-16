import logging
import torch
from torch import nn
from backbone.linears import SimpleContinualLinear
import time
import torch.nn.functional as F


def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    if 'tuna' not in name:
        raise NotImplementedError("Unknown backbone type: {}".format(name))

    from backbone import vit_tuna
    from easydict import EasyDict
    tuning_config = EasyDict(
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=16,
        d_model=768,
        _device=args["device"][0],
        vpt_on=False,
        vpt_num=0,
    )
    if name == "vit_base_patch16_224_tuna":
        model = vit_tuna.vit_base_patch16_224_adapter(
            num_classes=args["nb_classes"], global_pool=False,
            drop_path_rate=0.0, tuning_config=tuning_config, r=args["r"]
        )
    elif name == "vit_base_patch16_224_in21k_tuna":
        model = vit_tuna.vit_base_patch16_224_in21k_adapter(
            num_classes=args["nb_classes"], global_pool=False,
            drop_path_rate=0.0, tuning_config=tuning_config, r=args["r"]
        )
    else:
        raise NotImplementedError("Unknown backbone: {}".format(name))
    return model


class DAFNet(nn.Module):
    """Dynamical Adapter Fusion Network."""

    def __init__(self, args, pretrained):
        super().__init__()
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = 768
        self.fc = None
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, nb_classes):
        if self.fc is None:
            self.fc = SimpleContinualLinear(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=False)

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        return self.backbone(x, adapter_id, train, fc_only)

    # ------------------------------------------------------------------ #
    #  Fisher computation                                                  #
    # ------------------------------------------------------------------ #

    def _compute_fisher_and_gradients(self, train_loader, current_task_id, known_classes):
        """Compute diagonal FIM and gradient vector for the current task adapter."""
        logging.info("[DAF] Computing Fisher matrix...")
        st = time.time()

        current_adapter = self.backbone.cur_adapter
        was_training = self.training
        self.eval()

        fisher_acc = {n: torch.zeros_like(p.data) for n, p in current_adapter.named_parameters()}
        grad_acc = {n: torch.zeros_like(p.data) for n, p in current_adapter.named_parameters()}
        logsoft = nn.LogSoftmax(dim=1)
        sample_count = 0

        for batch_data in train_loader:
            if len(batch_data) == 3:
                _, inputs, targets = batch_data
            else:
                inputs, targets = batch_data
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = inputs.size(0)
            self.zero_grad()

            features = self.backbone(inputs, adapter_id=current_task_id, train=True)["features"]
            logits = self.fc(features)["logits"][:, known_classes:]  # current-task classes only
            loss = -F.nll_loss(logsoft(logits), targets - known_classes, reduction='none')
            exp_cond_prob = torch.mean(torch.exp(loss.detach()))
            torch.mean(loss).backward()

            for name, param in current_adapter.named_parameters():
                if param.grad is not None:
                    fisher_acc[name] += exp_cond_prob * param.grad.data.clone().pow(2) * batch_size
                    grad_acc[name] += param.grad.data.clone() * batch_size
            sample_count += batch_size

        fisher_matrix = {n: fisher_acc[n] / sample_count for n in fisher_acc}
        gradient_matrix = {n: grad_acc[n] / sample_count for n in grad_acc}

        logging.info(f"[DAF] Fisher computed in {round(time.time() - st, 2)}s over {sample_count} samples.")
        if was_training:
            self.train()
        return fisher_matrix, gradient_matrix

    # ------------------------------------------------------------------ #
    #  After-task fusion                                                   #
    # ------------------------------------------------------------------ #

    def after_task_training(self, train_loader, current_task_id, known_classes):
        """Compute Fisher/gradients and perform DAF fusion after each task."""
        logging.info(f"[DAF] Fusing after task {current_task_id}...")

        fisher_matrix, gradient_matrix = self._compute_fisher_and_gradients(
            train_loader, current_task_id, known_classes
        )

        # Perform DAF fusion
        self.backbone.merge(gradient_matrix, fisher_matrix)
