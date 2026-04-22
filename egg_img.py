# Upstream EGGROLL ships ASCII-art banners (EGG_IMG, CHICK_IMG) printed at
# startup. `tasks.py` imports them unconditionally at module import time; other
# modules (including `es_lora_multinode.py`) transitively pull that import.
# We don't use the banners anywhere in our ESvPG fork, so we ship empty
# strings to satisfy the import without changing behavior.
EGG_IMG = ""
CHICK_IMG = ""
