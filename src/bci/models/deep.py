"""Neural-network models (braindecode) wrapped as ordinary sklearn classifiers.

The wrapper builds the network inside `fit`, once it has seen the data's shape, so the model
factories stay zero-argument like every other model in `bci.models`. Everything the network
learns from data stays inside `fit` and therefore inside the training fold (RULES.md rule 1):

- per-channel scaling (mean/std) is computed from the training trials only;
- early stopping (deciding when to stop training) watches a validation slice carved out of the
  training trials, never the fold's test trials.

An "epoch" in this file means one pass of training over all training trials (deep-learning
usage), not an EEG epoch/trial.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit

ARCHITECTURES = ("eegnet", "shallow")


def _build(arch: str, n_chans: int, n_outputs: int, n_times: int) -> torch.nn.Module:
    import braindecode.models as M

    if arch == "eegnet":
        # EEGNet (Lawhern et al. 2018): a compact CNN with temporal filters, then per-filter spatial
        # filters over electrodes, then separable convolutions.
        return M.EEGNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    if arch == "shallow":
        # ShallowConvNet (Schirrmeister et al. 2017): temporal conv, spatial conv, squaring and
        # log-average pooling, a learned version of band-power features.
        return M.ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, final_conv_length="auto")
    raise ValueError(f"unknown architecture {arch!r}; choose from {ARCHITECTURES}")


class BraindecodeClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        arch: str = "eegnet",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        max_epochs: int = 150,
        patience: int = 30,
        val_fraction: float = 0.2,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.arch = arch
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed
        self.verbose = verbose

    def _scale(self, X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((X - self.mean_) / self.std_).astype(np.float32))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # Inner train/validation split, from the training trials only.
        inner = StratifiedShuffleSplit(n_splits=1, test_size=self.val_fraction, random_state=self.seed)
        tr, va = next(inner.split(np.zeros(len(y_idx)), y_idx))

        # Per-channel scaling from the inner training trials (EEG is tens of µV; networks train
        # best on values around 1).
        self.mean_ = X[tr].mean(axis=(0, 2), keepdims=True)[0]
        self.std_ = X[tr].std(axis=(0, 2), keepdims=True)[0] + 1e-6
        Xt, Xv = self._scale(X[tr]), self._scale(X[va])
        yt, yv = torch.from_numpy(y_idx[tr]), torch.from_numpy(y_idx[va])

        self.net_ = _build(self.arch, X.shape[1], len(self.classes_), X.shape[2])
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        best_loss, best_state, best_epoch, waited = np.inf, None, 0, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            self.net_.train()
            order = rng.permutation(len(tr))
            for start in range(0, len(order), self.batch_size):
                b = order[start : start + self.batch_size]
                if len(b) < 2:  # batch norm needs at least 2 samples
                    continue
                opt.zero_grad()
                loss = loss_fn(self.net_(Xt[b]), yt[b])
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                out = self.net_(Xv)
                val_loss = float(loss_fn(out, yv))
                val_acc = float((out.argmax(1) == yv).float().mean())
            self.history_.append({"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc})
            if val_loss < best_loss - 1e-4:
                best_loss, best_state, best_epoch, waited = val_loss, copy.deepcopy(self.net_.state_dict()), epoch, 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        if self.verbose:
            print(f"  {self.arch}: stopped at epoch {epoch}, best epoch {best_epoch}, inner val loss {best_loss:.3f}")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.net_.eval()
        with torch.no_grad():
            out = torch.cat([self.net_(self._scale(X[i : i + 256])) for i in range(0, len(X), 256)])
        return self.classes_[out.argmax(1).numpy()]


def eegnet(seed: int = 0, **kwargs) -> BraindecodeClassifier:
    params = dict(arch="eegnet", lr=1e-3, weight_decay=0.0, seed=seed)
    return BraindecodeClassifier(**{**params, **kwargs})


def shallow_convnet(seed: int = 0, **kwargs) -> BraindecodeClassifier:
    # Learning rate and weight decay as recommended in braindecode's examples for this network.
    params = dict(arch="shallow", lr=0.0625 * 0.01, weight_decay=0.0, seed=seed)
    return BraindecodeClassifier(**{**params, **kwargs})
