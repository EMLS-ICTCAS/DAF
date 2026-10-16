import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import DAFNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from torch.distributions.multivariate_normal import MultivariateNormal

num_workers = 8


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, s=20, m=0.0):
        super().__init__()
        self.s = s
        self.m = m
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels):
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat(
            [torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = DAFNet(args, True)
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]

        for n, p in self._network.backbone.named_parameters():
            if 'adapter' not in n and 'head' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable:,} trainable parameters.')

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task

        self._network.update_fc(self._total_classes - self._known_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.backbone.to(self._device)
        self._network.fc.to(self._device)
        optimizer = self._get_optimizer(self._network.backbone)
        scheduler = self._get_scheduler(optimizer)

        self._init_train(train_loader, test_loader, optimizer, scheduler)
        self._network.backbone.adapter_update()
        self._compute_mean(self._network.backbone)
        if self._cur_task > 0:
            self._network.after_task_training(
                train_loader=train_loader,
                current_task_id=self._cur_task,
                known_classes=self._known_classes,
            )
            self._classifier_align(self._network.backbone)

    def _get_optimizer(self, model):
        base_params = {'params': [p for n, p in model.named_parameters() if 'adapter' in n and p.requires_grad],
                       'lr': self.init_lr, 'weight_decay': self.weight_decay}
        fc_params = {'params': self._network.fc.parameters(),
                     'lr': self.init_lr, 'weight_decay': self.weight_decay}
        return optim.SGD([base_params, fc_params], momentum=0.9)

    def _get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr
            )
        return None

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        loss_fn = AngularPenaltySMLoss(s=self.args["scale"], m=self.args["m"])
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            losses, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                features = self._network.backbone(inputs, adapter_id=self._cur_task, train=True)["features"]
                logits = self._network.fc(features)["logits"]
                loss = loss_fn(logits[:, self._known_classes:], targets - self._known_classes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task, epoch + 1, self.args['tuned_epoch'],
                losses / len(train_loader), train_acc,
            )
            prog_bar.set_description(info)
        logging.info(info)

    @torch.no_grad()
    def _compute_mean(self, model):
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1), source="train", mode="test", ret_data=True,
            )
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size * 3, shuffle=False, num_workers=4)
            vectors = []
            for _, _inputs, _targets in idx_loader:
                _vectors = model(_inputs.to(self._device), adapter_id=self._cur_task, train=True)["features"]
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)
            self.cls_mean[class_idx] = vectors.mean(dim=0).to(self._device)
            cov = torch.cov(vectors.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device)
            if self.args["ca_storage_efficient_method"] == 'variance':
                cov = torch.diag(cov)  # store only the diagonal
            self.cls_cov[class_idx] = cov

    def _classifier_align(self, model):
        for p in self._network.fc.parameters():
            p.requires_grad = True

        network_params = [{'params': self._network.fc.parameters(),
                           'lr': self.ca_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        task_size = abs(self._known_classes - self._total_classes)
        self._network.eval()
        prog_bar = tqdm(range(self.crct_epochs))
        for epoch in prog_bar:
            sampled_data, sampled_label = [], []
            num_sampled_pcls = self.batch_size * 5
            for class_idx in range(self._total_classes):
                if self.args["decay"]:
                    t_id = class_idx // task_size
                    decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                    mean = torch.tensor(self.cls_mean[class_idx], dtype=torch.float64).to(self._device) * (0.9 + decay)
                else:
                    mean = self.cls_mean[class_idx].to(self._device)
                cov = self.cls_cov[class_idx].to(self._device)
                if self.args["ca_storage_efficient_method"] == 'variance':
                    cov = torch.diag(cov)
                try:
                    m = MultivariateNormal(mean.float(), cov.float())
                    samples = m.sample(sample_shape=(num_sampled_pcls,))
                except (ValueError, RuntimeError):
                    samples = None
                if samples is None or not torch.isfinite(samples).all():
                    # Classes with very few samples give rank-deficient covariances whose
                    # fp32 Cholesky factor can be non-finite; resample in fp64.
                    m = MultivariateNormal(mean.double(), cov.double())
                    samples = m.sample(sample_shape=(num_sampled_pcls,)).float()
                sampled_data.append(samples)
                sampled_label.extend([class_idx] * num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            sf_indexes = torch.randperm(sampled_data.size(0))
            inputs = sampled_data[sf_indexes]
            targets = sampled_label[sf_indexes]

            losses, correct, total = 0.0, 0, 0
            for _iter in range(self._total_classes):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                logits = self.args['scale'] * self._network.fc(inp)["logits"]
                loss = F.cross_entropy(logits, tgt)
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task, epoch + 1, self.crct_epochs,
                losses / self._total_classes, ca_acc,
            )
            prog_bar.set_description(info)
        logging.info(info)

    def _eval_cnn(self, loader):
        self.topk = 1
        self._network.eval()
        # Task 0 has no global adapter yet: evaluate with the first task adapter.
        # Afterwards, inference uses only the single global (merged) adapter.
        adapter_id = 0 if self._cur_task == 0 else self._cur_task + 2
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            with torch.no_grad():
                features = self._network.backbone(inputs, adapter_id=adapter_id, train=False)["features"]
                logits = self._network.fc(features)["logits"][:, :self._total_classes] * self.args['scale']
            predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)
