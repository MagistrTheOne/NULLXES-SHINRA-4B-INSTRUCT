# NULLXES SHINRA CORE
#
# Decoder-only Transformer — Language Intelligence Layer
# Model: NULLXES SHINRA-4B  (~3.93B parameters, tied embeddings)

from architecture.param_count import ShinraSpec, count_parameters

SHINRA_4B = ShinraSpec()
SHINRA_4B_COUNTS = count_parameters(SHINRA_4B)

__all__ = ["SHINRA_4B", "SHINRA_4B_COUNTS", "ShinraSpec", "count_parameters"]
