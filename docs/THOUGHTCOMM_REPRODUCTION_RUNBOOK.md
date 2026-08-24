# ThoughtComm external-repository runbook

This document records the local reproduction task, the failure mode we hit, the
runtime-only repair, and the cluster/A100 handoff.

## Task and scope

The goal is to compare the upstream
`PhucInAI/EXPERIMENT.2026.ThoughtComm` implementation with our
`latent_comm/` implementation using the cached Qwen3-1.7B checkpoint and the
MATH collection artifacts.

The upstream algorithm source is intentionally unchanged. The tracked changes
are limited to:

- `configs/local_qwen3_exp1.yaml`: local model/resource wiring;
- `scripts/run_adapter_2gpu_runtime.py`: runtime loader and numerical/backend
  controls, without changing the upstream objective or training loop;
- `scripts/build_context_resource.py`: deterministic resource-path
  preprocessing;
- `scripts/submit_adapter_public_a100.sbatch`: cluster job template.

Model weights, states, masks, caches, logs, and adapter checkpoints are not
tracked in git.

## Completed stages

The following local artifacts are complete:

- 500 examples, 3 agents, 2 rounds collected across physical GPUs 1 and 2;
- merged states: `(1000, 6144)`;
- merged contexts: 3000 records;
- 50-epoch autoencoder checkpoint;
- structure mask `(3, 1024)`.

## Original failure

The upstream adapter run used the full merged contexts and reached `40/1000`
states before a CUDA device-side assert. Earlier states already showed
`loss=nan`. The final error was raised during top-p sampling, but the sampling
error was downstream of a corrupted gradient/prefix state.

The relevant upstream code is `scripts/train_adapter.py`:

- prefix construction and direct injection: lines 237–248;
- semantic/NLL objective: lines 250–261;
- AdamW update without clipping or finite checks: lines 263–266.

The diagnostic evidence was:

1. The random initial prefix had mean L2 norm about `29.9`, while the Qwen3
   input embedding row mean L2 was about `1.54` (`19.4x` scale mismatch).
2. A direct Qwen3 control showed that an exactly zero prefix has finite forward
   logits but NaN prefix gradients; small nonzero prefixes have finite direct
   gradients.
3. Long-context eager float32 attention exceeded the available 46 GiB cards;
   the full 4096-token path tried to allocate another 272 MiB with only about
   232 MiB free.

## Runtime repair

The validated runtime settings are:

```text
model dtype                 float32
attention                   eager
prefix final-layer scale    0.005 (nonzero)
adapter learning rate       1e-5
gradient clipping            max_norm=1
top-p probability math      float32
cosine/NLL reductions       float32
hidden extraction           equivalent model.last_hidden_state path
```

These settings preserve the Eq.(11–12) quantities and the upstream training
loop. They address numerical scale, precision, and backend behavior only.

## Resource-truncated variant

The full 4096-token eager path cannot fit on the local L40S allocation. The
current practical variant truncates only the text resource to 1024 tokenizer
tokens; all 1000 states and all 3 agent records per state remain present.

Recreate it on another machine with:

```bash
python scripts/build_context_resource.py \
  --source_dir artifacts/full_qwen3_exp1/merged \
  --output_dir artifacts/full_qwen3_exp1/contexts_1024 \
  --model_path "$THOUGHTCOMM_MODEL_PATH" \
  --max_tokens 1024
```

This is a resource-truncated engineering run, not an exact full-context paper
reproduction. Keep that distinction in any report.

## Direct local run

Install the upstream requirements plus `omegaconf`, then run from the external
repository root:

```bash
export THOUGHTCOMM_MODEL_PATH=/path/to/Qwen3-1.7B
export THOUGHTCOMM_MODEL_DTYPE=float32
export THOUGHTCOMM_GRAD_CLIP=1
export THOUGHTCOMM_PREFIX_INIT_SCALE=0.005
export THOUGHTCOMM_ADAPTER_LR=1e-5
export THOUGHTCOMM_STABLE_SAMPLING=1
export THOUGHTCOMM_STABLE_LOSS=1
export THOUGHTCOMM_CONTEXTS_PATH=artifacts/full_qwen3_exp1/contexts_1024/contexts.jsonl
export THOUGHTCOMM_STATES_PATH=artifacts/full_qwen3_exp1/contexts_1024/states.pt
export THOUGHTCOMM_OVERRIDES='model.attn_implementation=sdpa'

CUDA_VISIBLE_DEVICES=0,1 python scripts/run_adapter_2gpu_runtime.py
```

For Sol, the validated full-context setup is two A100s on one node.
The short one-card smoke uses `THOUGHTCOMM_SINGLE_GPU=1`; the full run leaves
that variable empty so the launcher splits layers across logical CUDA devices 0 and 1.
Request two A100s with `-G a100:2`; do not request separate nodes. Verify the
cluster GPU resource syntax first.

## Sol/A100 handoff

ASU's current documentation says that the `public` partition uses a 7-day wall
time limit, and public GPU jobs should use the `public` QoS. It documents
`-G a100:1` for the common 80 GiB A100 request. See the official
[resource request guide](https://docs.rc.asu.edu/requesting-resources/),
[partition/QoS guide](https://docs.rc.asu.edu/partitions-and-qos/), and
[Sol hardware table](https://docs.rc.asu.edu/supercomputer-hardware/).

The tracked `scripts/submit_adapter_public_a100.sbatch` requests two
80 GiB A100s with `-p public -q public -N 1 -G a100:2` and runs one adapter epoch per
job. The unchanged upstream loop defaults to `adapter.grad_accum=8` and
retains long-context autograd graphs across states. The one-A100 short smoke
passed with `max_new_tokens=2`, but the first full-context run OOMed on the
first state at the real 32-token continuation length. Use native SDPA plus
`adapter.grad_accum=1` via `THOUGHTCOMM_OVERRIDES`; the full merged context
is the default resource, and twenty epochs use the checkpoint chain below.

Before submission:

1. Clone/pull this branch on Sol.
2. Copy the non-git artifacts: `ae/ae.pt`, `structure/mask.pt`, and the merged
   or resource-truncated `states.pt`/`contexts.jsonl`.
3. Make the Qwen snapshot available locally and set `THOUGHTCOMM_MODEL_PATH`.
4. Activate an environment containing PyTorch, Transformers, Accelerate,
   OmegaConf, Transformers-compatible Qwen3 support, and tqdm.
5. Check `sinfo`/`scontrol` for the site's current A100 availability. For this
   job use the documented multi-GPU syntax `-G a100:2`; do not replace it with
   an unverified GRES name.
6. Submit a short debug smoke first. ASU documents `debug` QoS as the fast
   syntax/path check with a 15-minute limit:

```bash
export THOUGHTCOMM_CONTEXTS_PATH=artifacts/full_qwen3_exp1/adapter_smoke/contexts.jsonl
export THOUGHTCOMM_STATES_PATH=artifacts/full_qwen3_exp1/adapter_smoke/states.pt
export THOUGHTCOMM_SINGLE_GPU=1
export THOUGHTCOMM_OVERRIDES='model.attn_implementation=sdpa adapter.epochs=1 adapter.continuation_max_new_tokens=2 adapter.grad_accum=1'
sbatch -p public -q debug -t 15 -G a100:1 scripts/submit_adapter_public_a100.sbatch
```

7. After the smoke succeeds, submit the one-epoch chain without putting
   passwords or tokens in the script:

```bash
export THOUGHTCOMM_MODEL_PATH=/cluster/path/Qwen3-1.7B
export PYTHON_BIN=/cluster/path/venv/bin/python
unset THOUGHTCOMM_SINGLE_GPU
export THOUGHTCOMM_OVERRIDES='model.attn_implementation=sdpa adapter.epochs=1 adapter.grad_accum=1'
unset THOUGHTCOMM_CONTEXTS_PATH THOUGHTCOMM_STATES_PATH THOUGHTCOMM_ADAPTER_CKPT
export THOUGHTCOMM_EPOCH=1
prev=""
for ep in $(seq 1 20); do
  export THOUGHTCOMM_EPOCH="${ep}"
  export THOUGHTCOMM_OUTPUT_DIR="/scratch/${USER}/thoughtcomm/adapter/epoch${ep}"
  if [[ "${ep}" -gt 1 ]]; then
    export THOUGHTCOMM_ADAPTER_CKPT="/scratch/${USER}/thoughtcomm/adapter/epoch$((ep-1))/adapter.pt"
  else
    unset THOUGHTCOMM_ADAPTER_CKPT
  fi
  if [[ -n "${prev}" ]]; then
    jid=$(sbatch --parsable --dependency="afterok:${prev}" --export=ALL scripts/submit_adapter_public_a100.sbatch)
  else
    jid=$(sbatch --parsable --export=ALL scripts/submit_adapter_public_a100.sbatch)
  fi
  echo "epoch ${ep}: job ${jid}"
  prev="${jid}"
done
```

Each chained job reloads the previous adapter weights and starts a fresh AdamW
optimizer; this is a weight-continuation protocol, not a bit-exact optimizer
state resume. If exact AdamW continuation is required, the upstream trainer
must additionally persist optimizer state at epoch boundaries.

The job writes `logs/thoughtcomm-adapter_<jobid>.out` and `.err` plus checkpoints under the
configured output directory. Monitor with `squeue`, `sacct`, and `seff`; ASU's
job-statistics guide notes that live GPU usage is best inspected on the running
compute node, while `sacct`/`seff` are for completed jobs.

Use `/scratch` for the model cache, working artifacts, and checkpoints while
the chain runs; ASU documents it as temporary, unbacked-up storage subject to
age-based cleanup. Copy the final adapter, resolved settings, and report to
`/home` or project `/data` when complete. See the
[storage guide](https://docs.rc.asu.edu/storage/) and
[scratch policy](https://docs.rc.asu.edu/scratch/).

No cluster credentials are required by the script itself; use the cluster's
normal SSH/Git credential mechanism. ASU recommends batch scripts for long
non-interactive work; see the [SBATCH guide](https://docs.rc.asu.edu/slurm-sbatch/).

## Current local status

The local resource-truncated full run is running in tmux session
`latentcomm-adapter-1024-full-v2` on physical GPUs 1 and 2. Its log is
`artifacts/full_qwen3_exp1/adapter_train_1024_eager_v2.log`.

The 2-state, 32-token validation completed successfully before this full run.
The full 20-epoch run has not yet produced an epoch checkpoint; do not treat it
as complete until `adapter_epoch1.pt` and subsequent checkpoints appear and
the final evaluation is run.
