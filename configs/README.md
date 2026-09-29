# SHINRA-4B configuration lives in this directory.

Active V2 path:
# architecture_v2.yaml        ARCHITECTURE ONLY (frozen geometry, no recipes)
# runtime_g4.yaml             G4 runtime/execution ONLY (not a train recipe)
# storage_g4.yaml             rolling drum rules/paths/disk limits (KEEP)
# colab.yaml                  Google Colab G4 hardware gate (KEEP, max_tokens: 0)
# stages/d_en_ru_pilot.yaml   Stage D EN/RU from C489 (BLOCKED until verified)
# stages/s0_bringup.yaml      S0 history (quarantine, not a source)
# stages/c_edu_en_pilot.yaml  Phase C history (reference only)
# stages/s1_language.yaml     legacy synth-v0 (quarantine, unauthorized)
# stages/s2_semantic.yaml     legacy (quarantine)

Deleted in cleanup (see git history; function replaced):
# shinra_4b.yaml              -> architecture_v2.yaml + runtime_g4.yaml + stage
# pretrain_a100.yaml / sft_a100.yaml / dpo_a100.yaml
#                             -> archived training/*.py (explicit --train-config required)
# pretrain_colab_100m.yaml    -> leftover hypothesis recipe (no replacement)
# data_mix.yaml / dataset_pilot.yaml -> data/specs/V2_UPGRADE_DATA_SPEC.md
# tokenizer.yaml              -> tokenizer/special_tokens.py (ID canon) + V2 spec §4 (acceptance)
# accelerate_a100*.yaml       -> removed earlier (no multinode plan)
