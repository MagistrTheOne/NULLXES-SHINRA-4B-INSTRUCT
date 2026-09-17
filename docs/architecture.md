# Архитектура SHINRA CORE

Полная спецификация: [`architecture/design.md`](../architecture/design.md).

Кратко: dense decoder-only 3.93B, RMSNorm, GQA 24/8, RoPE θ=1e6, SwiGLU 9216, QK-norm, Z-loss, HF `ShinraForCausalLM`.

Проверка счёта параметров (meta-device, без аллокации 4B весов):

```bash
PYTHONPATH=. python -m scripts.verify_architecture
PYTHONPATH=. python -m architecture.param_count
```
