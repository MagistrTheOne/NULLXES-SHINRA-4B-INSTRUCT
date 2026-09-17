# SHINRA runtime — inference engines, not training

vLLM / SGLang / TokenSpeed **не обучают** модель. Это serving после весов.

```
TRAINING  PyTorch FSDP FlashAttention
    ↓
WEIGHTS   ShinraForCausalLM + tokenizer DNA
    ↓
INFERENCE vLLM | SGLang | TokenSpeed | OpenAI HTTP
```

Контракт загрузки: `AutoModelForCausalLM` + `trust_remote_code`.  
HTTP: OpenAI-compatible `inference/server.py`.  
Long context: train 8k / arch 32k / YaRN later 128k.  
Tools: JSON `<|tool_call|>` / `<|tool_response|>` в tokenizer DNA.

## Когда какой движок

| Движок | Роль для SHINRA |
|--------|-----------------|
| `inference/server.py` | Phase 0 / Colab: наш FastAPI |
| vLLM | throughput, PagedAttention, continuous batching — после native plugin |
| SGLang | agent + structured decode + tool chains |
| TokenSpeed | latency-oriented serving |

Кастомный `ShinraForCausalLM` **не** встанет в vLLM как Llama. Пока нет `runtime/vllm/shinra_model.py` plugin, прод-путь: Transformers generate / наш HTTP.

## v1

Text-only. Vision encoder, million-token context и multimodal tokens — ветка **SHINRA-M** (CERBER → adapter), не текущий pretrain.
