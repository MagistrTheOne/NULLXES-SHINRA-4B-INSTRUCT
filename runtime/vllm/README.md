# vLLM + SHINRA

PagedAttention / continuous batching требуют зарегистрированный model class.

Архитектура: `nullxes_shinra` / `ShinraForCausalLM`. Пока нет plugin в `vllm.model_executor.models`:

```bash
bash runtime/vllm/serve.sh $CKPT
# → inference.server
```
