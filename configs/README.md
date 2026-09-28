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
# colab.yaml                  Google Colab G4 (RTX PRO 6000 Blackwell)
# pretrain_colab_100m.yaml    leftover hypothesis recipe; init uses shinra_4b.yaml
