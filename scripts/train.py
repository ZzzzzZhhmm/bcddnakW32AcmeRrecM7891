import hydra
import sys
import torch
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    try:
        run_training(cfg)
    finally:
        # Avoid masking the factual root exception with elastic-launcher/NCCL
        # teardown noise and release the process group on clean exits too.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as error:  # pragma: no cover - distributed teardown
                # Cleanup must never replace the training exception that
                # carries the actionable producer boundary and sample ID.
                print(
                    f"WARNING: distributed process-group cleanup failed: {error}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
