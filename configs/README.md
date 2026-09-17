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
# colab.yaml                  Google Colab Pro A100 40GB bring-up (not RTX 2080)
