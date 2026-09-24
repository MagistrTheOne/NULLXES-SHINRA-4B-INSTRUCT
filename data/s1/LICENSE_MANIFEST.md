# S1 license manifest

Training status here is the V1 builder decision. It is not a new license grant.

## TyDi QA

- dataset: TyDi QA
- upstream URL: https://github.com/google-research-datasets/tydiqa
- HF repo: https://huggingface.co/datasets/google-research-datasets/tydiqa
- license: apache-2.0
- license evidence: Hub dataset card field `license: apache-2.0`
- training status: APPROVED for the train split only
- attribution requirement: Apache-2.0 notice
- share-alike requirement: no
- restrictions/notes: validation is not a training split
- decision: APPROVED

## RussianSuperGLUE

- dataset: RussianSuperGLUE, configs DaNetQA, TERRa, RCB, RuCoS, RWSD, MuSeRC
- upstream URL: https://russiansuperglue.com/
- HF repo: https://huggingface.co/datasets/RussianNLP/russian_super_glue
- license: mit
- license evidence: Hub dataset card field `license: mit`
- training status: APPROVED for each config's train.jsonl only
- attribution requirement: MIT notice
- share-alike requirement: no
- restrictions/notes: validation and test stay out of train. The current `datasets` release cannot execute the repository script, so V1 reads the published zip train files directly.
- decision: APPROVED

## Held out of V1 train

### PAWS / PAWS-X

- dataset: PAWS, PAWS-X
- upstream URL: https://github.com/google-research-datasets/paws
- HF repo: https://huggingface.co/datasets/google-research-datasets/paws-x
- license: other
- license evidence: Hub card `license: other`; upstream license text is not copied here
- training status: LEGAL_REVIEW
- attribution requirement: not accepted for V1
- share-alike requirement: unknown until the upstream text is fixed
- restrictions/notes: not a training source in V1
- decision: LEGAL_REVIEW

### XNLI

- dataset: XNLI
- upstream URL: https://github.com/facebookresearch/XNLI
- HF repo: https://huggingface.co/datasets/xnli
- license: not stated on the Hub card
- license evidence: card has no license field
- training status: LEGAL_REVIEW / HOLD
- attribution requirement: unresolved
- share-alike requirement: unresolved
- restrictions/notes: do not train until provenance is closed
- decision: LEGAL_REVIEW

### Eval only

- XQuAD, Belebele, RuBQ 2.0, MMLU, FLORES
- decision: EVAL_ONLY
- training status: not used
- notes: benchmark or test-only cards. S0.5, S0.7, and S1 QA01–QA12 are diagnostic blacklists, not sources.
