# EXPERIMENT.2026.ThoughtComm
Implements the THOUGHTCOMM pipeline from *Thought Communication in Multiagent Collaboration* (NeurIPS 2025):
- Extract latent thoughts \(\hat Z_t\) via a sparsity-regularized autoencoder (Eq. 6–7).
- Recover thought–agent structure via Jacobian sparsity pattern \(B(J_{\hat f})\) and agreement \(\alpha_j\) (Eq. 8–10).
- Route thoughts per agent with agreement-based reweighting.
- Inject routed thoughts via prefix adaptation (Eq. 11–12) using a **brief continuation** objective to enforce fluency/coherence.

<!-- This version focuses on **engineering robustness** (reproducibility, caching, parallelism, checkpoints) **without changing the paper methodology**.

---

## Key upgrades vs the earlier scaffold

1. **True Eq.(12) adapter training** (differentiable):
   - For each agent context, generate a short continuation with prefix (y_gen) and without prefix (y_ref),
   - Optimize: `1 - cos(mean_hidden(y_gen), mean_hidden(y_ref))  - log p(y_gen | context, prefix)`
   - Uses *contextual token embeddings (last-layer hidden states)* as \(\bar\phi(\cdot)\) so gradients flow through the prefix.

2. **Parallel multi-agent runner (multi-GPU)**:
   - One process per agent (rank==agent_id),
   - all_gather states/messages, rank0 builds prefixes, scatter prefixes back.

3. **Reproducibility & run artifacts**:
   - Saves resolved config, git-like run id, logs, and checkpoints to `artifacts/<run_id>/`.

4. **Caching**:
   - Prompt-hash cache for baseline reference continuations (y_ref) to avoid regenerating.

5. **Memory & performance knobs**:
   - Optional bitsandbytes 8-bit/4-bit load, gradient checkpointing, torch.compile toggle.

6. **Tests** (smoke tests) and richer debug scripts. -->

---

## Install

```bash
# python -m venv .venv && source .venv/bin/activate
# pip install -U pip
# pip install -r requirements.txt



# Python 3.10
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Offline / local weights:
```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

---

## Quickstart (single GPU, sequential agents)

```bash
python -m thoughtcomm.cli.run   --config configs/hw/single_gpu_sequential.yaml   model.path=/assets/modelsQwen3-1.7B-Instruct   task.name=gsm8k task.max_examples=3
```

---

## Multi-GPU (parallel agents)

One process per agent:
```bash
accelerate launch --num_processes 3   -m thoughtcomm.cli.run   --config configs/hw/multi_gpu_parallel.yaml   model.path=/models/Llama-3-8B-Instruct   agents.num_agents=3 task.name=math task.max_examples=3
```

---

## Training pipeline (paper-style)

### 1) Collect states + contexts (baseline debate, no latent comm)
```bash
python scripts/collect_states.py --config configs/default.yaml   model.path=/models/Qwen3-1.7B-Instruct   task.name=gsm8k task.max_examples=500   output_dir=artifacts/data_gsm8k_qwen1p7b
```

Produces:
- `states.pt` : (N*T, nh) concatenated model states (last token per agent per round)
- `contexts.jsonl` : per (example, round, agent) prompt+response context for adapter training

### 2) Train autoencoder (Eq. 7)
```bash
python scripts/train_autoencoder.py --config configs/default.yaml   data.states_path=artifacts/data_gsm8k_qwen1p7b/states.pt   output_dir=artifacts/ae_gsm8k_qwen1p7b
```

### 3) Fit structure mask B(J) (Eq. 3–4 notion, estimated from decoder Jacobian)
```bash
python scripts/fit_structure.py --config configs/default.yaml   ae.ckpt_path=artifacts/ae_gsm8k_qwen1p7b/ae.pt   data.states_path=artifacts/data_gsm8k_qwen1p7b/states.pt   output_dir=artifacts/struct_gsm8k_qwen1p7b
```

### 4) Train prefix adapter (Eq. 12)
```bash
python scripts/train_adapter.py --config configs/default.yaml   model.path=/models/Qwen3-1.7B-Instruct   ae.ckpt_path=artifacts/ae_gsm8k_qwen1p7b/ae.pt   mask_path=artifacts/struct_gsm8k_qwen1p7b/mask.pt   contexts_path=artifacts/data_gsm8k_qwen1p7b/contexts.jsonl   output_dir=artifacts/adapter_gsm8k_qwen1p7b
```

---

## Debugging / interpretability

- `scripts/debug_latents.py`: agreement histogram + top-|z| dims.
- `scripts/latent_concepts.py`: top-k contexts for a latent dim.
- `scripts/ablate_latent.py`: zero/perturb selected latent dims and measure response change.

---

## Notes

- Paper uses prefix token count m=1 in main experiments.
- This repo defaults to `prefix_len=1` and `adapter.continuation_max_new_tokens=32`.
