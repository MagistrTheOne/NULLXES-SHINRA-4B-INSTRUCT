# SHINRA-4B configuration lives in this directory.
#
# shinra_4b.yaml              architecture v2 + unfrozen optimizer/FSDP recipe
# pretrain_a100.yaml          stage 1 recipe (not frozen)
# sft_a100.yaml               stage 2 recipe (not frozen)
# dpo_a100.yaml               stage 3 recipe (not frozen)
# accelerate_a100.yaml        8-GPU node (not frozen)
# accelerate_a100_multinode.yaml
# tokenizer.yaml
# data_mix.yaml
# dataset_pilot.yaml          SHINRA-COLAB-PILOT
# colab.yaml                  Google Colab G4 hardware (not a train recipe)
# stages/c_edu_en_pilot.yaml  Phase C 8M FineWeb-Edu EN from S0
# stages/s1_language.yaml     S1 230M
# stages/s2_semantic.yaml     S2 750M
# storage_g4.yaml             rolling disk ceiling
# pretrain_colab_100m.yaml    leftover hypothesis recipe; not the S0 drum
