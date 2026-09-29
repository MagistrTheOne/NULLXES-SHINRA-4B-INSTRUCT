# SHINRA-4B configuration lives in this directory.

# architecture_v2.yaml        ARCHITECTURE ONLY (frozen geometry, no recipes)
# runtime_g4.yaml             G4 runtime/execution ONLY (not a train recipe)
# shinra_4b.yaml              LEGACY mixed file: geometry + A100/FSDP recipe (do not use as recipe source)
# pretrain_a100.yaml          stage 1 recipe (not frozen, A100 archive, explicit opt-in only)
# sft_a100.yaml               stage 2 recipe (not frozen)
# dpo_a100.yaml               stage 3 recipe (not frozen)
# accelerate_a100.yaml        8-GPU node (not frozen)
# accelerate_a100_multinode.yaml
# tokenizer.yaml
# data_mix.yaml
# dataset_pilot.yaml          SHINRA-COLAB-PILOT
# colab.yaml                  Google Colab G4 hardware (not a train recipe)
# stages/c_edu_en_pilot.yaml  Phase C 8M FineWeb-Edu EN from S0 (reference only)
# stages/d_en_ru_pilot.yaml   Stage D EN/RU pilot from C489 (BLOCKED until C489 verified, review only)
# stages/s1_language.yaml     S1 230M (legacy synth-v0, quarantine)
# stages/s2_semantic.yaml     S2 750M (legacy, quarantine)
# storage_g4.yaml             rolling disk ceiling (KEEP: drum rules/paths/limits)
# pretrain_colab_100m.yaml    leftover hypothesis recipe; not the S0 drum (delete candidate, separate commit)
#
# DELETE CANDIDATES (separate commit, after link migration per docs/CONFIG_LINK_MAP.md):
# shinra_4b.yaml, sft_a100.yaml, old stages/pilots. Do NOT delete untracked user files,
# weights, source data, tokenizer artifacts, S0/C receipts.
