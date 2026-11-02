#!/usr/bin/env python
"""GPU truth test used for both SCOPE environments."""

from __future__ import print_function

import gc
import platform
import sys
import time
import traceback

failures = []


def step(name, fn):
    print("\n=== {} ===".format(name), flush=True)
    started = time.time()
    try:
        value = fn()
        print("PASS: {!r}".format(value), flush=True)
    except Exception as exc:
        failures.append((name, repr(exc)))
        print("FAIL: {!r}".format(exc), flush=True)
        traceback.print_exc()
    print("elapsed_seconds={:.6f}".format(time.time() - started), flush=True)


print("python={}".format(sys.version.replace("\n", " ")))
print("platform={}".format(platform.platform()))

try:
    import torch
except Exception:
    traceback.print_exc()
    sys.exit(2)

print("torch={}".format(torch.__version__))
print("torch.version.cuda={}".format(torch.version.cuda))
print("cudnn={}".format(torch.backends.cudnn.version()))
print("cuda.is_available={}".format(torch.cuda.is_available()))
print("cuda.device_count={}".format(torch.cuda.device_count()))

if torch.cuda.is_available():
    step("GPU name", lambda: torch.cuda.get_device_name(0))
    step("compute capability", lambda: torch.cuda.get_device_capability(0))
    step(
        "CUDA tensor creation and elementwise kernel",
        lambda: (torch.arange(4096, device="cuda", dtype=torch.float32) * 2.0).sum().item(),
    )
    step(
        "CUDA matrix multiplication",
        lambda: torch.mm(
            torch.randn(512, 512, device="cuda"),
            torch.randn(512, 512, device="cuda"),
        ).sum().item(),
    )

    def conv_test():
        layer = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).cuda()
        sample = torch.randn(8, 3, 64, 64, device="cuda")
        result = layer(sample)
        return (tuple(result.shape), float(result.sum().item()))

    step("CUDA Conv2d", conv_test)
    step("CUDA synchronize", torch.cuda.synchronize)

    def allocation_release_test():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(0)
        allocation = torch.empty(
            64 * 1024 * 1024, dtype=torch.float32, device="cuda"
        )
        allocation.fill_(1.0)
        torch.cuda.synchronize()
        during = torch.cuda.memory_allocated(0)
        del allocation
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated(0)
        if during - before < 256 * 1024 * 1024:
            raise RuntimeError("expected at least 256 MiB of measured allocation")
        if after != before:
            raise RuntimeError(
                "allocation was not released: before={} after={}".format(
                    before, after
                )
            )
        return {"before": before, "during": during, "after": after}

    step("CUDA 256 MiB allocation and release", allocation_release_test)
    print("memory_allocated_before_release={}".format(torch.cuda.memory_allocated(0)))
    print("memory_reserved_before_release={}".format(torch.cuda.memory_reserved(0)))
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.synchronize()
    except Exception:
        traceback.print_exc()
    print("memory_allocated_after_release={}".format(torch.cuda.memory_allocated(0)))
    print("memory_reserved_after_release={}".format(torch.cuda.memory_reserved(0)))
else:
    failures.append(("CUDA availability", "torch.cuda.is_available() is False"))

print("\nfailed_steps={}".format(len(failures)))
for failed_name, failed_error in failures:
    print("failed: {}: {}".format(failed_name, failed_error))
sys.exit(1 if failures else 0)
