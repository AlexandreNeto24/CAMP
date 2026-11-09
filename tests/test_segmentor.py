"""Segmentor package surface. Does not construct a model, download weights, or train."""

import inspect

from camp.models.segmentor import PretrainedUNet
from camp.train.segmentor import SegmentorTrainer, train_segmentor


def test_segmentor_symbols_import():
    assert isinstance(PretrainedUNet, type)
    assert isinstance(SegmentorTrainer, type)
    assert callable(train_segmentor)


def test_fit_and_train_segmentor_share_parameter_names():
    fit_sig = inspect.signature(SegmentorTrainer.fit)
    train_sig = inspect.signature(train_segmentor)
    fit_names = [name for name in fit_sig.parameters if name != "self"]
    train_names = [name for name in train_sig.parameters if name != "model"]
    assert fit_names == train_names
    assert list(fit_sig.parameters)[0] == "self"
    assert list(train_sig.parameters)[0] == "model"
    for name in fit_names:
        assert fit_sig.parameters[name].default == train_sig.parameters[name].default
        assert fit_sig.parameters[name].kind == train_sig.parameters[name].kind
