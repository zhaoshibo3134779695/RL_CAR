#!/usr/bin/env python
"""Minimal proof of the official SO-SCOPE return-value mismatch."""

from __future__ import print_function

import os
import sys

import torch

REPO_SCRIPTS = "/home/jovyan/work/research/scope_repro/repos/scope/scripts"
CHECKPOINT = (
    "/home/jovyan/work/research/scope_repro/repos/scope/model/so_scope_model.pth"
)
sys.path.insert(0, REPO_SCRIPTS)

from model import so_scope  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")
model = so_scope(input_channels=1, latent_dim=512, output_channels=1).to(device)
checkpoint = torch.load(CHECKPOINT, map_location=device)
print("checkpoint_keys={}".format(sorted(checkpoint.keys())))
load_result = model.load_state_dict(checkpoint["model"])
print("load_state_dict={}".format(load_result))
model.eval()
sample = torch.zeros(1, 10, 1, 64, 64, dtype=torch.float32)
with torch.no_grad():
    returned = model(sample)
print("returned_type={}".format(type(returned).__name__))
print("returned_shape={}".format(tuple(returned.shape)))
try:
    prediction, kl_loss = returned
    print("unexpected_unpack_success={} {}".format(prediction.shape, kl_loss.shape))
except Exception as error:
    print("official_decode_unpack_error={!r}".format(error))
    raise
