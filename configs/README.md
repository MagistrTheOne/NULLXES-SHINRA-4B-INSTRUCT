# SHINRA-4B production configuration lives in this directory.
#
# shinra_4b.yaml              model + optimizer + FSDP + A100 batch
# pretrain_a100.yaml          stage 1
# sft_a100.yaml               stage 2
# dpo_a100.yaml               stage 3
# accelerate_a100.yaml        8-GPU node
# accelerate_a100_multinode.yaml
# tokenizer.yaml
# data_mix.yaml
# dataset_pilot.yaml          SHINRA-COLAB-PILOT 5B spec / 100M session
# colab.yaml                  Google Colab Pro A100 80GB
# pretrain_colab_100m.yaml    hypothesis 100M tokens
