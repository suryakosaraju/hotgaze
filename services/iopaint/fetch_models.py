"""Download the model weights into the image at build time.

The service then runs with HF_HUB_OFFLINE=1 and never downloads while handling a
request. Files land under $XDG_CACHE_HOME, the same layout `iopaint start
--model-dir` reads: torch/hub/checkpoints for LaMa, huggingface for RMBG-1.4.
"""

from huggingface_hub import hf_hub_download
from iopaint.download import cli_download_model

cli_download_model("lama")
print(hf_hub_download("briaai/RMBG-1.4", "model.pth"))
