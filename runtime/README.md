# SHINRA runtime — inference engines, not training

vLLM / SGLang / TokenSpeed **не обучают** модель. Это serving после весов.

```
TRAINING  PyTorch FSDP FlashAttention
    ↓
WEIGHTS   ShinraForCausalLM + tokenizer DNA
    ↓
INFERENCE vLLM | SGLang | TokenSpeed | OpenAI HTTP
```

## Что взято у [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

Проверено на Hub (`pipeline_tag: image-text-to-text`, `model_type: qwen3_5`):

- это **VLM**: vision encoder + hybrid Gated DeltaNet / Gated Attention
- `rope_parameters.mrope_section: [11, 11, 10]`, `mrope_interleaved: true`
- native context 262144, YaRN override до 1M **на сервере**
- `--hf-overrides` в vLLM/SGLang/TokenSpeed — runtime RoPE, не датасет

SHINRA v1 **не копирует** MRoPE, vision tokens, million-token context, Gated DeltaNet.

Берём только:

- AutoModelForCausalLM + `trust_remote_code` как контракт загрузки
- OpenAI-compatible API
- YaRN как **будущий** long-context override (train 8k / arch 32k / later 128k)
- JSON tool calls, которые уже в tokenizer DNA

## Когда какой движок

| Движок | Роль для SHINRA |
|--------|-----------------|
| `inference/server.py` | Phase 0 / Colab: наш FastAPI |
| vLLM | throughput, PagedAttention, continuous batching — после native plugin |
| SGLang | agent + structured decode + tool chains |
| TokenSpeed | latency-oriented serving, тот же HF override контракт |

Кастомный `ShinraForCausalLM` **не** встанет в vLLM как Llama. Пока нет `runtime/vllm/shinra_model.py` plugin, прод-путь: Transformers generate / наш HTTP. Команды ниже — целевой контракт, не фейковый kernel.

## Чего нет в v1

MRoPE, vision encoder, 1M context, multimodal tokens. Это ветка **SHINRA-M** (CERBER encoder → SHINRA adapter), не текущий pretrain.
