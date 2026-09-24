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

## PAWS-Wiki

- dataset: PAWS-Wiki labeled_final
- upstream URL: https://github.com/google-research-datasets/paws/blob/master/LICENSE
- distribution README: https://github.com/google-research-datasets/paws/blob/master/README.md
- HF repo: https://huggingface.co/datasets/google-research-datasets/paws
- config: labeled_final
- split: train only
- license: Google PAWS dataset license, plus Wikipedia sentence text under CC BY-SA 3.0
- license evidence: upstream `LICENSE` states the dataset may be freely used for any purpose and asks for acknowledgement of Google LLC. Wikipedia source text is not relicensed by that file, so CC BY-SA 3.0 attribution and share-alike still apply to the sentences. The Apache-2.0 header on `qqp_generate_data.py` is a code license and is not the dataset license.
- training status: APPROVED for labeled_final train only
- attribution requirement: acknowledge Google LLC; attribute Wikipedia contributors and retain CC BY-SA share-alike on Wikipedia-derived sentence text
- share-alike requirement: yes, for the Wikipedia sentence text
- restrictions/notes: validation, test, unlabeled_final, and labeled_swap are not used. English only. A transformed training row is PUBLIC-DERIVED and keeps the upstream row id.
- decision: APPROVED

### PAWS-QQP

- dataset: PAWS-QQP
- upstream URL: https://github.com/google-research-datasets/paws/blob/master/README.md
- license: Quora Question Pairs terms, not the PAWS dataset license
- license evidence: the PAWS README states raw PAWS-QQP cannot be redistributed because of the QQP license. Quora's dataset release allows non-commercial use under its terms.
- training status: not approved
- redistribution: no
- decision: RESTRICTED

### PAWS-X

- dataset: PAWS-X
- upstream URL: https://github.com/google-research-datasets/paws
- HF repo: https://huggingface.co/datasets/google-research-datasets/paws-x
- training status: not a P0 source
- restrictions/notes: no Russian split. Not added to the train registry.
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
