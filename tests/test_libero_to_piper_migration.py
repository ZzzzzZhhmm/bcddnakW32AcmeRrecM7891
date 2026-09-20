import unittest

import torch

from fastwam.real.libero_to_piper import (
    LiberoToPiperMigrationError,
    migrate_fastwam_checkpoint_payload,
    migrate_proprio_encoder_state,
)


class LiberoToPiperMigrationTests(unittest.TestCase):
    def test_shrinks_8d_head_without_padding_piper(self):
        weight = torch.arange(4096 * 8, dtype=torch.float32).reshape(4096, 8)
        bias = torch.arange(4096, dtype=torch.float32)
        migrated = migrate_proprio_encoder_state({"weight": weight, "bias": bias})
        self.assertEqual(tuple(migrated["weight"].shape), (4096, 7))
        self.assertTrue(torch.equal(migrated["weight"][:, :6], weight[:, :6]))
        self.assertTrue(
            torch.allclose(
                migrated["weight"][:, 6], 0.5 * (weight[:, 6] + weight[:, 7])
            )
        )
        self.assertTrue(torch.equal(migrated["bias"], bias))

    def test_rejects_already_7d_or_warm_payload(self):
        with self.assertRaises(LiberoToPiperMigrationError):
            migrate_proprio_encoder_state(
                {
                    "weight": torch.zeros(4096, 7),
                    "bias": torch.zeros(4096),
                }
            )
        with self.assertRaises(LiberoToPiperMigrationError):
            migrate_fastwam_checkpoint_payload(
                {
                    "mot": {},
                    "proprio_encoder": {
                        "weight": torch.zeros(4096, 8),
                        "bias": torch.zeros(4096),
                    },
                    "warm_source": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
