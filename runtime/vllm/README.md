# vLLM + SHINRA

PagedAttention / continuous batching need a registered model class.

SHINRA architecture is `nullxes_shinra` / `ShinraForCausalLM`, not Llama/Qwen. Copying Qwen3.8 `--hf-overrides` with `mrope_interleaved` and `mrope_section: [11,11,10]` is invalid: those axes are vision/time, which SHINRA v1 does not have.

Until `vllm.model_executor.models` plugin lands:

```bash
bash runtime/vllm/serve.sh $CKPT
# → inference.server
```
