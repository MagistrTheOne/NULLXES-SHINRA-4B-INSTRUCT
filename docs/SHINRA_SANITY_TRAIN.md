# SHINRA — SANITY TRAIN → FOUNDATION MIX → IDENTITY

## Role and operating boundary

This document is the working policy for the repository `NULLXES-SHINRA-4B-INSTRUCT`.

The VS Code agent may write code, configs, validators, and runbooks. It may not launch GPU jobs or access Google Colab/G4 directly.

The operational rule is fixed:

> Agent writes code/config/validators/commands. Agent does not run Colab/GPU jobs. All heavy execution is performed manually by the user on the G4 runtime.

This document is intentionally staged:

1. Phase A — preflight
2. Phase B — dataset validation
3. Phase C — sanity train, only FineWeb-Edu
4. Phase D — sanity acceptance criteria
5. Phase E — resume test
6. Phase F — foundation mix only after sanity pass
7. Phase G — provenance manifest
8. Phase H — SHINRA identity stage
9. Phase I — identity evaluation
10. Phase J — required output and manual commands

---

## Hard guardrails

- No production train before sanity PASS.
- No 22B token launch before sanity run is valid.
- No rewrite of model architecture or tokenizer DNA without a blocker and explicit approval.
- No silent config mutation after an error. If a failure happens, the agent must produce a diagnosis and patch proposal, but the user decides whether to apply it and rerun.
- No automatic launch of Google Colab jobs.
- The true runtime source of truth is `torch.cuda.get_device_name()` from the actual G4 session, not the Colab UI.
- Do not touch the Hugging Face repo used for model publishing or any model weights repository. Use dataset IDs only, and keep the local tokenizer artifact and local configs authoritative.
- All heavy runtime launches are manual, not agent-executed.

---

## Machine bootstrap and init policy

The user runs the machine setup manually, and the agent must never auto-execute this step.

Minimal environment bootstrap:

```bash
cd /content/NULLXES-SHINRA-4B-INSTRUCT
python -V
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=$PWD
```

Runtime environment assertions:

```bash
python - <<'PY'
import sys, torch
print(sys.version)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device_name', torch.cuda.get_device_name(0))
    print('compute_capability', torch.cuda.get_device_capability(0))
    print('total_vram', torch.cuda.get_device_properties(0).total_memory)
PY
```

The repository itself is not modified as a model-publishing repo. The HF repository is not a source of truth for the training runtime.

---

## Canonical SHINRA contract

The repo already defines the canonical architecture and tokenizer contract.

Architecture values must match:

```text
vocab_size = 131072
hidden_size = 2560
layers = 36
attention_heads = 32
kv_heads = 8
head_dim = 128
intermediate_size = 9728
context = 32768
rope_theta = 1e6
```

Token ID contract must match:

```text
UNK = 0
BOS = 1
EOS/EOT = 2
PAD = 3
END_OF_TEXT = 18
```

Critical invariant:

```text
EOS != END_OF_TEXT
```

The repo also defines the G4 runtime profile in `configs/colab.yaml`.

---

## Phase A — preflight

### Purpose

Run a single automatic preflight before training begins. This must block the trainer if the environment or model contract is invalid.

### Required runtime checks

```text
Python
PyTorch
CUDA
GPU name
compute capability
total VRAM
Transformers
Accelerate
Datasets
Tokenizers
Safetensors
PyArrow
```

### Required checks for SHINRA contract

The validator must assert:

```python
assert vocab_size == 131072
assert hidden_size == 2560
assert num_hidden_layers == 36
assert num_attention_heads == 32
assert num_key_value_heads == 8
assert head_dim == 128
assert intermediate_size == 9728
assert max_position_embeddings == 32768
assert rope_theta == 1_000_000.0
```

If any mismatch occurs:

```text
PREFLIGHT = FAIL
TRAIN = BLOCKED
```

### Required tokenizer checks

The preflight must validate the tokenizer artifact and enforce the invariant:

```python
assert tokenizer.vocab_size == 131072
assert tokenizer.unk_token_id == 0
assert tokenizer.bos_token_id == 1
assert tokenizer.eos_token_id == 2
assert tokenizer.pad_token_id == 3
assert tokenizer.convert_tokens_to_ids("<|end_of_text|>") == 18
assert tokenizer.eos_token_id != tokenizer.convert_tokens_to_ids("<|end_of_text|>")
```

Also check:

- `encode(text, add_special_tokens=False)` works
- tokenizer does not add BOS automatically
- packer adds exactly one BOS
- each document ends with exactly one `END_OF_TEXT`
- token IDs remain within model vocab
- tokenizer artifact has fixed revision/hash

Failure policy:

```text
PREFLIGHT = FAIL
TRAIN = BLOCKED
```

### Example preflight shell stub

```bash
python - <<'PY'
import json, os, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

print('python_ok')
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
print('device_count', torch.cuda.device_count())
print('cuda_version', torch.version.cuda)

root = Path('.').resolve()
for p in [root / 'tokenizer' / 'artifacts', root / 'configs']:
    print('exists', p, p.exists())
PY
```

---

## Phase B — dataset auto-check

### Purpose

Validate each candidate dataset before training, but do not start the training job. This is an automatic data gate, not a training stage.

### Source list

Canonical candidate sources:

```text
HuggingFaceFW/fineweb-edu
mlfoundations/dclm-baseline-1.0
HuggingFaceFW/fineweb-2
```

For RU FineWeb2, use the real Russian metadata after checking the actual HF card and config. Do not guess. If the correct subset is `rus_Cyrl`, use that only after validation.

### Dataset metadata to validate

For each source, the validator must inspect:

```text
repo_id
revision
subset/config
split
license
schema
text column
sample count if available
streaming availability
```

### Sample validation

Take a small sample and validate:

- `text` exists
- type is string
- empty strings are absent or filtered
- absurdly short records are counted and tracked
- no tokenizer special tokens already embedded in raw text
- UTF-8 is clean
- EN is genuinely EN, RU is genuinely RU
- no HTML-only junk
- no repeated document terminators
- tokenization passes
- token IDs remain valid for the SHINRA vocab

### Tokenizer compatibility check

Run a sample through the packer contract:

```text
raw
→ tokenizer(add_special_tokens=False)
→ BOS
→ body
→ END_OF_TEXT
→ pack
```

The validator must assert that the resulting sequence is valid and that final token IDs are `< vocab_size`.

### Dataset-gating example

```python
from datasets import load_dataset

def validate_text_record(row):
    text = row.get('text')
    assert isinstance(text, str)
    assert text.strip() != ''
    assert '<|end_of_text|>' not in text
    assert '<|bos|>' not in text
    return True

# small sample only; do not start training
sample = load_dataset('HuggingFaceFW/fineweb-edu', split='train[:2000]')
for row in sample:
    validate_text_record(row)
```

---

## Phase C — first sanity train

### Purpose

This is not a production run. This is a learning-mechanics sanity pass to show the model is valid, not to produce the final foundation model.

We use a single clean English source only:

```text
HuggingFaceFW/fineweb-edu
```

No DCLM, no RU, no code, no math, no identity during the sanity stage.

### Sanity config

```yaml
stage: s0_learning_sanity

sequence_length: 2048
micro_batch_size: 1
gradient_accumulation_steps: 8
attention_implementation: sdpa
precision: bf16
shuffle: deterministic
seed: fixed
gradient_checkpointing: true
max_steps: 100
save_steps: 25
log_steps: 1
eval_steps: 25
```

### Exact tokens per update

```text
1 × 2048 × 8 = 16,384 tokens/update
```

For 100 steps:

```text
100 × 16,384 = 1,638,400 tokens
```

This is a minimal but meaningful learning run. It is enough to prove:

- loss moves
- gradients are finite
- optimizer steps succeed
- checkpoint saves work
- resume works
- throughput is sane
- VRAM is sane

### Manual sanity stage command stub

```bash
python scripts/fetch_tokenizer.py --dest tokenizer/artifacts
python -m training.trainer --config configs/stages/s0_learning_sanity.yaml
```

This should be adapted to the actual repo’s stage path and CLI after the user confirms the final config file name.

---

## Phase D — sanity acceptance

### Required PASS conditions

A sanity run passes only if all of the following are true:

```text
model loads
tokenizer verified
dataset verified
forward finite
loss finite
backward finite
gradients finite
optimizer.step succeeds
checkpoint saves
checkpoint reloads
resume succeeds
```

Additional checks:

```text
loss does not explode
grad_norm does not explode permanently
no recurring NaN
no recurring Inf
```

### PASS output style

```text
SANITY_PASS = TRUE
```

### Failure policy

If any of the above fails, the result is not a model verdict. It is a runtime diagnosis. The agent must report root cause and patch proposal only.

---

## Phase E — resume test

This is mandatory and is not optional extras.

### Required resume pattern

```text
step 0 → 25
save
STOP
```

Then user resumes manually:

```text
resume step 25
→ step 50+
```

### Things to validate at resume

```text
global_step
optimizer
scheduler
consumed_tokens
dataset_index
checkpoint hash
tokenizer revision
config hash
```

Expected output:

```text
RESUME_EXACT = PASS / FAIL
```

This is the strongest proof that the runtime path is stable.

---

## Phase F — foundation mix only after sanity PASS

A production-like foundation run is allowed only after the sanity run is accepted.

### Working foundation mix

```yaml
fineweb_edu_en: 0.55
dclm_en: 0.25
fineweb2_ru: 0.15
specialist: 0.05
```

However, the specialist bucket remains disabled until those sources are reviewed and approved individually.

Therefore, the first “real” production-like mix is effectively:

```text
FineWeb-Edu EN   57.9%
DCLM EN          26.3%
FineWeb2 RU      15.8%
```

if the first three buckets are normalized to 100%.

If the user does not approve the specialist sources, do not run the 5% bucket. Keep it reserved and unenabled.

---

## Phase G — dataset identity and provenance

Every production shard must have a clear provenance manifest.

Required metadata:

```text
dataset_repo
dataset_revision
dataset_config
split
source_hash
shard_hash
number_documents
number_tokens
tokenizer_revision
tokenizer_hash
packer_version
config_hash
created_at
```

Resume must refuse mismatched dataset revision or tokenizer revision unless there is an explicit override.

The policy is:

```text
No silent acceptance of a different dataset revision.
No silent acceptance of a different tokenizer revision.
```

---

## Phase H — SHINRA identity

Identity is not part of the foundation sanity run.

The identity layer is a separate, small dataset stage after the model can train cleanly.

### Identity metadata

```text
Name: SHINRA
Russian name: ШИНРА
Organization: NULLXES
Model family: SHINRA
```

### Identity dataset policy

Do not add “brand stuffing” to foundation pretraining.

Instead, create a small, curated identity dataset with varied prompts such as:

```text
Who are you?
What is your name?
Как тебя зовут?
Ты кто?
What model are you?
Как называется эта модель?
Who developed you?
К какой модели ты относишься?
```

Targets should vary in form but stay semantically stable:

```text
I am SHINRA, a language model developed by NULLXES.
```

and

```text
Я ШИНРА, языковая модель NULLXES.
```

Not a huge synthetic dump. A compact, high-quality identity set only.

---

## Phase I — identity evaluation

After the identity stage, run a frozen probe set and classify the result.

Probe values:

```text
SHINRA
ШИНРА
NULLXES
```

Check that the model:

- is not claiming to be Llama
- is not claiming to be GPT
- is not claiming to be Qwen
- is not claiming to be DeepSeek
- does not invent OpenAI / Meta / Alibaba identities
- consistently recognizes SHINRA / ШИНРА

Output:

```text
IDENTITY_PASS
IDENTITY_PARTIAL
IDENTITY_FAIL
```

---

## Phase J — exact user commands to run manually

The agent must not execute GPU jobs. The user runs these commands manually on Colab/G4.

### 1) Preflight

```bash
cd /content/NULLXES-SHINRA-4B-INSTRUCT
python - <<'PY'
import torch
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
print('device_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
print('torch=', torch.__version__)
print('cuda=', torch.version.cuda)
PY
```

### 2) Validate tokenizer and config

```bash
python scripts/fetch_tokenizer.py --dest tokenizer/artifacts
python - <<'PY'
from pathlib import Path
import json

root = Path('tokenizer/artifacts')
for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
    path = root / name
    print(name, path.exists())
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        print('keys', list(data.keys())[:10])
PY
```

### 3) Validate dataset sample

```bash
python - <<'PY'
from datasets import load_dataset

dataset = load_dataset('HuggingFaceFW/fineweb-edu', split='train[:2000]')
print(dataset)
for row in dataset:
    text = row.get('text')
    assert isinstance(text, str)
    assert text.strip()
    break
print('dataset-sample-ok')
PY
```

### 4) Start sanity train

```bash
python -m training.trainer --config configs/stages/s0_learning_sanity.yaml
```

Use a config file created specifically for the sanity stage.

### 5) Resume test

```bash
# after step 25 save
python -m training.trainer --config configs/stages/s0_learning_sanity.yaml --resume-from /path/to/step-00000025
```

### 6) Evaluate

```bash
python -m evaluation.harness --config configs/stages/s0_learning_sanity.yaml
```

### 7) Only after PASS: prepare foundation mix

```bash
# config template only, not a launch command yet
# likely a new stage YAML with:
# - fineweb_edu_en 0.55
# - dclm_en 0.25
# - fineweb2_ru 0.15
# - specialist reserved
```

---

## Package compatibility note

The environment may contain upgraded package families that are relevant for the runtime stack. The same package versions should be treated as compatibility information, not as a reason to change the SHINRA model architecture.

Observed/expected upgrade list relevant to the runtime:

```text
affine 2.4.0 -> 3.0.1
bigframes 2.42.0 -> 2.48.0
cryptography 49.0.0 -> 50.0.1
jax / jaxlib 0.7.2 -> 0.11.1
kagglesdk 0.1.23 -> 0.1.37
llvmlite 0.43 -> 0.44
ml_dtypes 0.5.4 -> 0.6.0
numba 0.60.0 -> 0.61.2
optree 0.19.1 -> 0.20.0
pandas 2.2.2 -> 2.2.3
stringzilla 4.6.2 -> 5.1.2
tokenizers 0.22.2 -> 0.23.1
xxhash 3.8.1 -> 4.0.1
```

This list is not a signal to restructure SHINRA. It is a compatibility note for the operational environment and should be checked during preflight if a runtime issue appears.

---

## Recommended first run concept

The correct first serious sanity pass is:

```text
Context: 2048
Micro batch: 1
Accumulation: 8
Precision: bf16
Attention: sdpa
Gradient checkpointing: true
Max steps: 100
Source: FineWeb-Edu only
```

This is intentionally minimal but still meaningful.

If the run passes preflight, dataloader validation, pack validation, forward/backward, checkpoint/reload, and resume, the repo becomes a real training-valid model stack instead of only an architecture stub.

The identity layer and production mix are then added as separate stages, not mixed into the foundation sanity run.

---

## Final working policy

The repo should follow this order:

1. preflight
2. dataset validation
3. sanity train on FineWeb-Edu only
4. resume test
5. sanity PASS gate
6. foundation mix stage
7. provenance manifest
8. SHINRA identity stage
9. identity eval

This sequence is strict. It prevents premature data sprawl, prevents brand pollution in the foundation corpus, and keeps the actual model behavior auditable.
