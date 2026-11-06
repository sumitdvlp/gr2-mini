# gr2-mini — re-implementing GR2 on a single 6 GB GPU

> **Paper:** *GR2: Generative Reasoning Re-ranker* — Liang, Li, Xu et al. (Meta AI), arXiv:2602.07774
> **Goal of this repo:** understand GR2 by rebuilding every stage of it from scratch, in small
> testable modules, at a scale that runs on a GTX 1660 Ti (6 GB) + 32 GB RAM on Windows.

---

## 1. What GR2 does (the 60-second version)

A recommender usually works in two steps: a **retriever** proposes a short list of candidate
items (here: 10), and a **re-ranker** re-orders that list so the item the user will actually buy
next ends up on top. GR2 makes the re-ranker an **LLM that reasons** ("this user bought a
shampoo and a conditioner, so the styling cream fits best...") before outputting the new order.

Three things make that work:

```
             ┌──────────────────────── STAGE 1: tokenized mid-training ────────────────────────┐
item text ──►│ text encoder ─► RQ-VAE ─► Semantic ID  e.g. <s_a_12><s_b_3><s_c_250><s_d_7>     │
             │                      new tokens added to the LLM vocabulary, LLM is trained     │
             │                      to "speak" SIDs (SID ⇄ title, history ⇄ next SID)          │
             └─────────────────────────────────────────────────────────────────────────────────┘
             ┌──────────────────────── STAGE 2: reasoning traces + SFT ────────────────────────┐
prompt  ────►│ big TEACHER LLM writes step-by-step explanations + a ranking                    │
(history +   │   targeted sampling (teacher sees the answer)  /  rejection sampling (it doesn't,│
10 cands)    │   keep only traces whose top-1 is correct) ─► SFT the student on those traces    │
             └─────────────────────────────────────────────────────────────────────────────────┘
             ┌──────────────────────── STAGE 3: reinforcement learning ────────────────────────┐
             │ DAPO (a GRPO variant): sample G answers per prompt, reward = how many positions │
             │ the true item moved UP (+ format bonus only when it helps → no reward hacking)  │
             └─────────────────────────────────────────────────────────────────────────────────┘
```

## 2. What "mini" changes — and why

Every *algorithm* in the paper is implemented. Only the *sizes* change.

| Component | Paper | gr2-mini | Why |
|---|---|---|---|
| Dataset | Amazon Beauty & Sports (2014, 5-core) | Amazon **Beauty** 5-core (same data) + a `tiny` subset for fast iteration | Same data, so we can compare against the paper's tables |
| Text encoder | not named | `all-MiniLM-L6-v2` (22 M params, 384-d) | Fast, good enough |
| SID tokenizer | RQ-VAE + EMA + k-means init + dead-code reset + diversity + contrastive + random last level | **all of it** | Fully faithful (tiny MLP, trains in minutes) |
| Retriever (pre-rank) | mid-trained Qwen3-8B | **SASRec** (small transformer) | An 8B retriever is impossible on 6 GB; SASRec has similar Recall@10 on Beauty |
| Student LLM | Qwen3-8B (full fine-tune) | **Qwen3-0.6B** in fp32 + LoRA + trainable SID-token embeddings | Fits in ~4.3 GB of VRAM |
| Teacher LLM | Qwen3-32B | Local model through Ollama (e.g. `qwen3:8b` 4-bit), or any OpenAI-compatible API | 32B can't run locally |
| RL | DAPO (verl-style, multi-GPU) | DAPO **written from scratch** in plain PyTorch | That's the most educational part! |

Expect smaller *absolute* numbers than the paper. What we want to reproduce are the
*qualitative* findings: EMA ≫ no-EMA for SID uniqueness, SFT alone can hurt Recall@1, RL fixes it,
rejection-sampled traces > targeted traces, unconditional format reward → reward hacking.

## 3. How to work through this repo

Every module folder `gr2mini/mXX_name/` contains:

| File | Who writes it | Purpose |
|---|---|---|
| `README.md` | me | The lesson: concepts, paper equations, pitfalls, hints |
| `*.py` | **you** | Stubs with signatures + docstrings (the *contract*). Bodies raise `NotImplementedError` |
| `tests/test_*.py` | me | Unit tests that check your implementation against the contract |
| `run.py` | **you** | A "try it" script that runs the module on real data (the live example) |

**The loop for each module:**

1. Read the module `README.md` (and the paper section it points to).
2. Implement the functions in the order the README suggests.
3. Run that module's tests: `python -m pytest gr2mini/m04_rqvae -v`
4. When everything is green, run the live example: `python -m gr2mini.m04_rqvae.run --config configs/tiny.yaml`
5. Then on the full data: `--config configs/beauty.yaml`
6. Tell me when you're stuck — ask for a **hint level**: (1) conceptual nudge, (2) pseudocode,
   (3) I review your code and point at the bug. I won't paste solutions unless you ask.

Please don't edit the tests to make them pass 🙂 — if you think a test is wrong, tell me; it might be!

## 4. Setup (Windows)

```powershell
# a project-local virtual env has already been created in .venv
.\.venv\Scripts\Activate.ps1          # (if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126   # already done
python -m pip install -r requirements.txt                                        # already done
python -m pytest gr2mini -q           # everything fails at first — that's your TODO list
```

Optional but recommended: `git init` and commit after every module goes green.

## 5. Roadmap

Do the modules in order — each one's output is the next one's input.

| # | Module | Paper | You build | Status |
|---|---|---|---|---|
| **Phase A — Foundations** |||||
| 00 | [m00_setup](gr2mini/m00_setup/) | — | config loading, seeding, GPU/dtype checks for your 1660 Ti | 🟢 ready |
| 01 | [m01_metrics](gr2mini/m01_metrics/) | App. G | Recall@K, NDCG@K, SID collision rate / uniqueness | 🟢 ready |
| **Phase B — Items → Semantic IDs** |||||
| 02 | [m02_data](gr2mini/m02_data/) | §5.1.1 | download Amazon Beauty, k-core, chronological sequences, splits, item text | 🟢 ready |
| 03 | [m03_text_embed](gr2mini/m03_text_embed/) | App. A Eq.18 | sentence embeddings of item text, with caching | 🟢 ready |
| 04 | [m04_rqvae](gr2mini/m04_rqvae/) | §2.1–2.3, App. A | RQ-VAE: k-means++ init, residual quantizer, EMA, dead codes, diversity + contrastive losses | 🟢 ready |
| 05 | [m05_semantic_ids](gr2mini/m05_semantic_ids/) | §2.3, Eq.10, 33–35 | assign SIDs, random last level, collision resolution, SID ⇄ token strings | 🟢 ready |
| **Phase C — Candidate generation** |||||
| 06 | [m06_retriever](gr2mini/m06_retriever/) | §5.1 (pre-rank) | SASRec retriever → top-10 candidate lists | 🟢 ready |
| **Phase D — Stage 1: tokenized mid-training** |||||
| 07 | [m07_llm](gr2mini/m07_llm/) | §2.4 | load Qwen3-0.6B, add SID tokens, LoRA, memory-efficient log-probs | 🟢 ready |
| 08 | [m08_midtrain](gr2mini/m08_midtrain/) | §2.4, App. B | item-alignment tasks + training loop | 🟢 ready |
| **Phase E — Stage 2: reasoning traces + SFT** |||||
| 09 | [m09_prompts](gr2mini/m09_prompts/) | §3.1–3.2, App. C–D | re-ranking chat prompts (targeted / rejection / policy) | 🟢 ready |
| 10 | [m10_parsing](gr2mini/m10_parsing/) | App. E | robust JSON → regex → validation output parser | 🟢 ready |
| 11 | [m11_rewards](gr2mini/m11_rewards/) | §4.3 Eq.14–15 | rank-improvement reward + conditional format reward | 🟢 ready |
| 12 | [m12_traces](gr2mini/m12_traces/) | §3, Eq.11–12 | teacher client, targeted & rejection sampling | 🟢 ready |
| 13 | [m13_sft](gr2mini/m13_sft/) | §4.2 Eq.13 | SFT with separate reasoning / output loss weights | 🟢 ready |
| **Phase F — Stage 3: RL** |||||
| 14 | [m14_dapo](gr2mini/m14_dapo/) | §4.3 Eq.16–17 | DAPO: group rollouts, advantages, dynamic sampling, clip-higher | 🟢 ready |
| **Phase G — Results** |||||
| 15 | [m15_eval_demo](gr2mini/m15_eval_demo/) | §5, App. F | evaluation tables (like Tables 5–6) + live re-ranking demo | 🟢 ready |

🟢 = README + stubs + tests ready (all modules, as of Phase C).

## 6. Data & artifact flow

```
data/raw/reviews_Beauty_5.json.gz ─┐
data/raw/meta_Beauty.json.gz ──────┴─ m02 ─► data/processed/items.jsonl, sequences.json
                                     m03 ─► artifacts/item_emb.npy               [N_items, 384]
                                     m04 ─► artifacts/rqvae.pt
                                     m05 ─► artifacts/sids.json                  item_id ⇄ (a,b,c,d)
                                     m06 ─► artifacts/candidates_{train,valid,test}.jsonl
                                     m08 ─► artifacts/midtrain_lora/
                                     m12 ─► artifacts/traces_{targeted,rejection}.jsonl
                                     m13 ─► artifacts/sft_lora/
                                     m14 ─► artifacts/rl_lora/
                                     m15 ─► results table + live demo
```

The `tiny` config writes to `data/processed_tiny/` and `artifacts/tiny/` so both scales coexist.

## 7. Your hardware — what to keep in mind

* **GTX 1660 Ti = Turing (7.5) without tensor cores.** fp16 matmuls are **8× slower than fp32** on it
  (measured), and there's no native bf16. Most LLM tutorials assume bf16/fp16. **We run everything in
  fp32** (see the m00 correction).
* **CUDA sysmem fallback:** on Windows, running out of VRAM silently spills into system RAM and makes
  training crawl, with no error. Watch `nvidia-smi`, or set the NVIDIA Control Panel "CUDA – Sysmem
  Fallback Policy" to "Prefer No Sysmem Fallback".
* **6 GB VRAM, but Windows uses ~1.5–2 GB of it for the display** (WDDM). Plan for ~4 GB.
  Close browsers or games while training.
* **Windows:** no vLLM, no flash-attention 2, and `DataLoader(num_workers>0)` needs the
  `if __name__ == "__main__":` guard. We use HF `generate`, SDPA attention, and `num_workers=0`.
* **Windows console encoding:** printing tokenizer output (e.g. `Ġ`) or product titles can crash with
  `UnicodeEncodeError: 'charmap' codec...`. Run `$env:PYTHONUTF8 = "1"` once per terminal, or add it to
  your PowerShell profile.
* Wall-clock budget on your machine (full Beauty), measured with my reference implementation:
  data + SIDs ≈ 5 min; SASRec ≈ 7 min; mid-training ≈ 1–3.5 h; teacher traces ≈ 1–3 h; SFT ≈ 30–45 min
  per trace type; DAPO ≈ 3 h per run; evaluation ≈ 1 h per checkpoint (300 test examples).

## 8. Where the paper is ambiguous — and what we do

Write-ups are never complete. These are the gaps I found and the choice we make for each
(each module README explains its own in more detail):

1. **EMA codebook (Eq. 4–5)** is written as an EMA over *optimizer-updated* weights. We use the
   standard VQ-VAE EMA (cluster-size-weighted running means, van den Oord et al. 2017 / Razavi et al. 2019),
   because the paper cites that method.
2. **Appendix A has no encoder/decoder MLP** (ĥ = Σq is compared to h directly). We use the TIGER-style
   RQ-VAE (MLP encoder → quantizer → MLP decoder), which is what "RQ-VAE" usually means.
3. **Contrastive loss (Eq. 2–3)**: what representation, which positives? We use the RQ-VAE latent,
   positives = items bought next to each other in a training sequence, negatives = in-batch items.
4. **Dead-code reset (Eq. 8–9)** resets with *the single* worst residual. With several dead codes at once,
   that would create duplicate codes, so we use the top-n worst residuals.
5. **Random last level (Eq. 10)** leaves ~0.02% collisions. An LLM needs a one-to-one SID ⇄ item map, so
   we re-draw the random level for colliding items until all SIDs are unique.
6. **Retriever = mid-trained 8B LLM.** We use SASRec. As in the paper, we keep only examples
   whose top-10 list contains the true item (otherwise the reward in Eq. 14 is undefined).
7. **Data leakage between retriever and re-ranker**: we use a 4-way split per user (see m02 README).
8. **Tables report Recall@9 although K = 10.** We report R@1, R@5, R@9, N@5, N@10 like the tables.
9. **Reward for unparseable outputs** isn't specified. We fall back to the pre-rank order (R_rank = 0, R_fmt = 0).
