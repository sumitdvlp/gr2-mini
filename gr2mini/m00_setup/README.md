# m00 — Setup: configs, seeds, and your GPU

**Paper:** nothing specific. This is the plumbing every later module uses.
**Time:** ~1 hour. **Files you implement:** `config.py`, `seed.py`, `device.py`, `check_env.py`.

## Why this module exists

ML experiments fail in boring ways. Two runs differ and you don't know why (seeds), you edit
the wrong config copy (no inheritance), or training NaNs after 2 hours because your GPU doesn't
really support the dtype a tutorial used (bf16 on a 1660 Ti). This module removes those
three failure modes before we start.

---

## 1. Configs with inheritance — `config.py`

We have one config per *scale*: `configs/beauty.yaml` (full) and `configs/tiny.yaml` (fast).
`tiny.yaml` only lists what's different and starts with `extends: beauty.yaml`.

```
beauty.yaml                     tiny.yaml                       load_config("tiny.yaml")
rqvae:                          extends: beauty.yaml            rqvae:
  codebook_size: 256      +     rqvae:                    =       codebook_size: 64   ← overridden
  epochs: 300                     codebook_size: 64               epochs: 60          ← overridden
  use_ema: true                   epochs: 60                      use_ema: true       ← inherited
```

The core is `deep_merge(base, override)`:
* dict + dict → merge recursively (that's why `use_ema` survives above)
* anything else → override wins. **Lists are replaced, not appended.** (`hidden_dims: [64]`
  in tiny should mean `[64]`, not `[256, 128, 64]`.)
* **Don't mutate the inputs.** A classic bug: you merge into `base` in place, and the next
  config that extends the same base inherits the first child's values. The tests check this.

`extends` paths are **relative to the YAML file**, not to your current directory. Hint:
`Path(path).parent / cfg["extends"]`.

**Command-line overrides.** To run an ablation (e.g. "RQ-VAE without EMA") you shouldn't have to
copy a YAML file. Every `run.py` accepts `--set key.path=value` (repeatable), and `apply_overrides`
applies them. Values are parsed with YAML, so `false`, `0.1`, `null`, and `[64, 32]` get the right types:

```powershell
python -m gr2mini.m04_rqvae.run --config configs/tiny.yaml --set rqvae.use_ema=false --set rqvae.epochs=20
```

## 2. Reproducibility — `seed.py`

Randomness comes from four independent generators: Python `random`, NumPy, PyTorch CPU, and
PyTorch CUDA (one per device). `seed_everything` seeds all of them.

Determinism on GPU is a separate thing. cuDNN picks the fastest kernel per shape, and some
fast kernels use atomics, which makes the order of float additions (and therefore the
result) vary run to run. `deterministic=True` trades speed for bit-exact repeats. It's useful
when debugging and unnecessary for training runs.

Later modules take an explicit `torch.Generator` for their random ops (k-means++ sampling, random
SID levels...). That keeps a module's randomness independent of whatever ran before it, so it
is testable in isolation. `make_generator` is a one-liner for that.

## 3. Your GPU — `device.py`

| Fact | Consequence |
|---|---|
| GTX 1660 Ti = **Turing, compute capability 7.5** | fp16 is fast. **bf16 is not native** (needs ≥ 8.0, Ampere). |
| `torch.cuda.is_bf16_supported()` may still return `True` | It counts *emulated* bf16 by default, which is slow and not what you want. Check `torch.cuda.get_device_capability()` instead. |
| 6 GB total, **~1.5–2 GB used by Windows for the display** | Budget ~4 GB. `torch.cuda.mem_get_info()` tells you the real free memory. |
| fp16 max value is **65 504** | Activations or logits beyond that become `inf`, then `NaN`. Large-vocab LLM logits are where this bites, so we keep a sanity check. |

`pick_dtype(device, prefer)` encodes those rules so later modules never hard-code a dtype:

```
cpu                        → float32  (always; fp16 on CPU is slow and pointless)
cuda, prefer="fp32"        → float32
cuda, prefer="fp16"        → float16
cuda, prefer="bf16"        → bfloat16 if capability >= (8, 0) else float16   ← your card falls back
```

`fp16_sanity_check` multiplies two random matrices in fp32 and in fp16 and compares the results.
On a healthy GPU the relative error is ~1e-3. If it's `inf`/`NaN` or large, train in fp32.

### Back-of-envelope VRAM math (you'll use this in m07)

For a model with P parameters:

| What | Bytes |
|---|---|
| weights in fp16 | 2·P |
| full fine-tuning with Adam (fp32 master weights + grads + 2 moments) | ~16·P |
| LoRA: frozen fp16 weights + tiny trainable adapters | ~2·P + (adapter · 16) |
| logits for one sequence of T tokens, vocab V, fp32 | 4·T·V |

Qwen3-0.6B: P ≈ 0.6 B → 1.2 GB of fp16 weights. Full fine-tuning would need ~10 GB (impossible), and LoRA
fits. Logits for T = 1500, V = 152k cost **0.9 GB per sequence**, which is why m07 computes the logits
only at the positions we need.

---

## Implementation order

1. `config.py`: `deep_merge`, then `load_config`, then `apply_overrides`.
2. `seed.py`: `seed_everything`, `make_generator`.
3. `device.py`: `get_device`, `pick_dtype`, `vram_info`, `fp16_sanity_check`.
4. `check_env.py`: glue that prints a report (not unit-tested).

## Test & try

```powershell
python -m pytest gr2mini/m00_setup -v
python -m gr2mini.m00_setup.check_env --config configs/tiny.yaml
```

Expected `check_env` output, roughly:

```
python 3.13.5 | torch 2.14.1+cu126 | cuda available: True
device: cuda | NVIDIA GeForce GTX 1660 Ti | capability (7, 5) | free 4.2 / 6.0 GB
dtype for prefer=bf16 -> torch.float16   (no native bf16 on this GPU)
fp16 sanity: ok=True  max_rel_err=0.0007
config: rqvae.codebook_size=64 (tiny overrides beauty's 256)
```
