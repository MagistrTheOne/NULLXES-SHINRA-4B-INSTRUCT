# S1-P0 Train Contract

Status: FROZEN FOR COLLECTION
Amendment: P0 Amendment 1 (2026-09-24)
Parent: S0.4.1 step 1
Does not modify S1_DATA_SPEC.md
Does not authorize training by itself

## Parent path

Published checkpoint:

`exp/stage0-2026-09-23/checkpoints/S0.4.1-parent`

Recorded state: stage S0.4.1, step 1, score 20/20, lr 1.5e-8.
Full weight. No adapters.
Architecture and tokenizer stay frozen.

## Budget

Supervised target tokens: 2,500,000.

The stop rule is the tokenizer-counted assistant target count, not the row count.
About 510,000 rows is only the estimate from the 20,000-row pilot, where 2,593,811 input tokens produced 97,790 target tokens.

Full 10–14M target tokens stay closed until P0 shows a task-conditioned answer instead of the identity attractor.

## Update

Target tokens per update: 32,768.
Expected updates: 2,500,000 / 32,768 = 76, remainder 1,232 tokens on the last update.

Loss is the sum of per-token cross-entropy over supervised target tokens, divided once by the actual target-token count of that update.
A microbatch mean must not be given weight 1.
z-loss 1e-5 uses the same supervised-token denominator.
Prompt tokens stay labeled -100.

`F.cross_entropy` default mean inside one forward is not this objective.
`data/pack.py` is not the P0 packer. It writes `attention_mask` of all ones across concatenated documents and copies those tokens into labels.

P0 packing may place several examples in one row only with isolated attention: a token attends to earlier tokens of the same example and to nothing else.
Pad tokens are invisible.
Cross-example attention fails the contract.

Sequence length is at most 8192.
Examples are not padded up to 8192 one by one.

## Optimizer

Implementation: `torch.optim.AdamW`, `foreach=False`.
Learning rate: 1.5e-8.
Betas: 0.9, 0.999.
eps: 1e-6.
Weight decay: 0.
Gradient clip: 1.0.
Precision: FP16.
`use_cache`: false.
Gradient checkpointing: on.
Seed: 20260924.

Seed 7407 and seed 42 are not this run.
The seed does not change after this contract.

## Data

Train is a later build.
It must reject every prompt hash in `data/s1/p0_dev/dev_prompt_hashes.json`.
It must reject the frozen S0.5, S0.7, and QA12 blacklist.
Identity replay is the complete validated unique train bank, used once.
The hard maximum is 6% of supervised target tokens.
The 5% identity target in S1_DATA_SPEC.md is the full-S1 curriculum target. It is not a minimum quota for this 2.5M run.
All ten families are required.
DEV and BLIND do not count toward the 2.5M.

## DEV

Frozen before train collection.

Path: `data/s1/p0_dev/s1_dev_v1.jsonl`
SHA256: `f5d1512e3877e969b05f4cbd1f95c021ec2c099db024defa733647076ae3fb23`
Records: 2000
EN 1000 / RU 1000
Families: 300, 300, 240, 240, 200, 240, 200, 100, 80, 100 for S1-01 through S1-10.
Unique prompts: 2000.
Index base 9000000, except identity, which uses a separate surface from the train identity bank.

S1 BLIND is not frozen here.
P0 cannot promote a checkpoint.
Promotion waits for a frozen S1 DEV report plus frozen QA12 and an identity regression.
Train loss is not a promotion metric.

## Gate

Before the first optimizer step, the manifest must record parent hash, this seed, TRAIN sha256, DEV sha256, input tokens, target tokens, target tokens per update, and the fraction of labels that are not -100.
The manifest must also record the effective P0 family targets from Amendment 1.

## P0 Amendment 1

Date: 2026-09-24
Reason: unique identity replay capacity

Previous operative rule, kept here and in git history: identity replay is 5% of supervised target tokens, hard maximum 6%.

That 5% floor is withdrawn for this P0 run only. S1_DATA_SPEC.md is unchanged. Full S1 stays on its own family table, including identity at 5%.

P0 is an acquisition experiment. Its parent, S0.4.1, already shows identity behavior. Identity rows in P0 are replay and regression protection, not the main thing this run is trying to teach. A target-token floor larger than the unique semantic bank would require duplicate prompts, cosmetic paraphrases, or longer identity answers. Those would make the corpus worse. P0 therefore plays the validated unique bank once and keeps the 6% ceiling.

S1-10 stays inside three facts: the name is SHINRA, the creator is NULLXES, and SHINRA is not NULLXES. No mission, personality, gender, consciousness, product category, biography, or new canonical fact.

Rules for the bank:

- use every validated unique P0 identity train prompt once
- duplicate prompts are forbidden
- do not mint cosmetic paraphrases to fill a quota
- do not lengthen the identity answer to fill a quota
- the actual fraction is whatever that bank tokenizes to
- 12,816 is the count from the 2026-09-24 validation, not a second authority. If a later validation of the same bank recounts, the recount replaces 12,816 and the family table below is recomputed

Validation on 2026-09-24, frozen tokenizer, each prompt once:

- unique prompts: 2,304 (1,152 EN and 1,152 RU)
- supervised target tokens: 12,816 (EN 6,480, RU 6,336)
- fraction of 2,500,000: 0.51264%

The released remainder is R = 2,500,000 - I. I is the validated bank. For I = 12,816, R = 2,487,184.

R is split across S1-01 through S1-09 by the original weights 15, 15, 12, 12, 10, 12, 10, 5, 4. Those weights sum to 95. Each budget is the integer quotient of R * weight / 95. Leftover tokens go by the largest-remainder method. Equal remainders break by family id. S1-10 is I, not a rescaled share.

Effective targets for the validated count:

- S1-01: 392,713
- S1-02: 392,713
- S1-03: 314,171
- S1-04: 314,171
- S1-05: 261,809
- S1-06: 314,171
- S1-07: 261,809
- S1-08: 130,904
- S1-09: 104,723
- S1-10: 12,816

Sum: 2,500,000. The total budget does not shrink. Atomic overshoot of the last accepted row is still allowed on top of this table and must be reported. The build manifest records the same table.
