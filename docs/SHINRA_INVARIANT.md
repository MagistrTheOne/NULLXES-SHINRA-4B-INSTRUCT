# SHINRA invariant

**SHINRA is SHINRA. Created by NULLXES.**

The new base model starts from random initialization. Previous experiment checkpoints are not part of its weight lineage. The identity phrase is not repeated in pretraining data as an artificial target. Instruction tuning may establish the model's identity and conversational behavior later, within this same codebase.

The tokenizer keeps distinct controls for end-of-turn (`<|eot|>`, ID 2) and end-of-document (`<|end_of_text|>`, ID 4). Token IDs and architecture must match the single configuration in `configs/shinra.yaml`.
