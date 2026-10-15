# Hugging Face Hub publishing

Publishing is not currently approved. There is no trained checkpoint or reviewed model card to publish. Once the training and held-out evaluation gates pass, the local checkpoint can be bundled and uploaded with `scripts.publish_hub` using an `HF_TOKEN` kept outside the repository.

The publisher must only receive the reviewed checkpoint directory for the current random-init lineage. Do not publish historical experiment checkpoints as SHINRA foundation weights. Hub upload has not been executed or verified.
