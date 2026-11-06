"""Try it:  python -m gr2mini.m00_setup.check_env --config configs/tiny.yaml

Prints a one-screen health report of your environment. Not unit-tested: it's glue code.
See the README for the expected output.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the gr2-mini environment.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    args = parser.parse_args()

    # TODO(m00):
    #   1. Print python version, torch version, torch.cuda.is_available().
    #   2. device = get_device(cfg["device"]); print vram_info(device) and
    #      torch.cuda.get_device_capability(device) if it's cuda.
    #   3. Print pick_dtype(device, "bf16") and explain the fallback if it isn't bfloat16.
    #   4. Print fp16_sanity_check(device).
    #   5. cfg = load_config(args.config); print cfg["rqvae"]["codebook_size"] to prove
    #      `extends` works (tiny -> 64, beauty -> 256).
    raise NotImplementedError("TODO(m00): implement check_env.main")


if __name__ == "__main__":
    main()
