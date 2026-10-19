"""Retired legacy tokenizer smoke cell; use the canonical Native tokenizer CLI."""

raise SystemExit(
    "UNAVAILABLE: this legacy Colab smoke script depended on an external-source corpus "
    "and launched pytest. Use `python -m tokenizer.train_tokenizer --corpus-dir <audited-NULLXES-corpus> "
    "--output-dir <new-candidate-dir>` after the synthetic corpus manifest is accepted."
)
