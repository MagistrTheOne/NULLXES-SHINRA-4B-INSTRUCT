# Публикация на Hugging Face Hub

Чекпоинт уже содержит `config.json` с `auto_map`, веса safetensors, токенизатор и исходники `configuration_shinra.py` / `modeling_*.py` (копируются при save).

```bash
export HF_TOKEN=hf_xxx
python -m scripts.export_hf_bundle --checkpoint outputs/shinra-4b-instruct/final/step-00008000
python -m scripts.publish_hub \
  --checkpoint outputs/shinra-4b-instruct/final/step-00008000 \
  --repo-id NULLXES/SHINRA-4B-INSTRUCT \
  --private
```

После публикации:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "NULLXES/SHINRA-4B-INSTRUCT",
    trust_remote_code=True,
    torch_dtype="bfloat16",
)
```

Model card: `docs/MODEL_CARD.md` копируется в `README.md` репозитория модели.

Корпус, рабочее состояние и опубликованные веса разведены в `docs/storage.md`. Локальный диск пользователя для датасетов и checkpoint не используется.
