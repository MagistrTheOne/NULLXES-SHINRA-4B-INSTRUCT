# Архитектура SHINRA CORE

Полная спецификация: [`architecture/design.md`](../architecture/design.md).

**v2 contract:** dense decoder-only **3,969,056,256** params, residual 2560, Q width 4096, GQA 32/8, 36 layers, SwiGLU 9728, RoPE θ=1e6, QK-norm, Z-loss, vocab 131072, HF `ShinraForCausalLM`. Tokenizer DNA не Qwen.

v1 (32 × 3072, GQA 24/8) закрыта. Веса не наследуются.

Проверка счёта параметров (meta-device, без аллокации 4B весов):

```bash
PYTHONPATH=. python -m scripts.verify_architecture
PYTHONPATH=. python -m architecture.param_count
```
