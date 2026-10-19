# SHINRA Pretraining Data Pipeline V2

The linked V2 pipeline is historical external-source documentation and is disabled for new training. Do not run its commands. The active policy uses only NULLXES Synthetic Engine; see the [synthetic native roadmap](ROADMAP_SYNTHETIC_NATIVE.md) and [static audit](AUDIT_SYNTHETIC_NATIVE.md).

New training records must come exclusively from the NULLXES Synthetic Engine. External text datasets and third-party LLM generated answers are not admitted. The legacy source acquisition modules remain in the repository for history, but are excluded from the installed CLI and active YAML policy. Historical data locations outside the inspected repository copy are unknown and were not deleted.

The current engine code defines deterministic arithmetic, device mode transition, inventory, and coordinate movement families with answer verifiers. It has not been executed. This does not establish corpus quality, broad language coverage, or semantic split integrity. See [the synthetic native roadmap](ROADMAP_SYNTHETIC_NATIVE.md) and [the static audit](AUDIT_SYNTHETIC_NATIVE.md).

The synthetic corpus assigns generator families before split and packs isolated documents. The Native SHINRA tokenizer is intended to be trained from synthetic train records only. No token inventory or runtime build has yet verified this contract.

`data.training_enabled`, `training.enabled` and `runtime.colab_launch_allowed` remain false. No runtime has been approved.
