# Google Colab launch gate

No Colab notebook or launcher is active. Google Colab G4 or a verified A100 is a future runtime target; the actual device, available memory, BF16 support, free disk, and model preflight estimate must be recorded before training.

The single configuration currently has `status: data_tokenizer_and_runtime_audit_required`, both training gates disabled, and `colab_launch_allowed: false`. `training.pretrain` rejects that state. Do not launch a machine until data and tokenizer audits are complete, the configuration is frozen, automated validation passes, and the review explicitly accepts readiness.
