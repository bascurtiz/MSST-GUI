"""Unit tests verifying Apollo architecture support and config normalization."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from utils.settings import load_config, get_model_from_config
from utils.model_utils import prefer_target_instrument, demix
from ml_collections import ConfigDict


class TestApolloSupport(unittest.TestCase):

    def test_load_config_apollo_uni_normalization(self):
        """Test that native Look2Hear config_apollo_uni.yaml is normalized with training/audio/inference."""
        config = load_config('apollo', 'configs/config_apollo_uni.yaml')
        self.assertIn('training', config)
        self.assertIn('audio', config)
        self.assertIn('inference', config)
        self.assertEqual(config.training.model_type, 'apollo')
        self.assertEqual(config.training.instruments, ['restored', 'addition'])
        self.assertEqual(config.training.target_instrument, 'restored')
        self.assertEqual(config.audio.sample_rate, 44100)
        self.assertEqual(config.audio.chunk_size, 132300)
        self.assertEqual(config.inference.batch_size, 1)

    def test_get_model_from_config_apollo(self):
        """Test that get_model_from_config instantiates Apollo model without _target_ TypeError."""
        model, config = get_model_from_config('apollo', 'configs/config_apollo_uni.yaml')
        self.assertEqual(model.__class__.__name__, 'Apollo')
        self.assertEqual(config.training.model_type, 'apollo')

    def test_prefer_target_instrument_safe(self):
        """Test prefer_target_instrument with normal, sparse, and empty configs."""
        cfg1 = ConfigDict({'training': {'instruments': ['restored', 'addition'], 'target_instrument': 'restored'}})
        self.assertEqual(prefer_target_instrument(cfg1), ['restored'])

        cfg2 = ConfigDict({'training': {'instruments': ['restored', 'addition']}})
        self.assertEqual(prefer_target_instrument(cfg2), ['restored', 'addition'])

        cfg3 = ConfigDict({'model_type': 'apollo'})
        self.assertEqual(prefer_target_instrument(cfg3), ['restored'])

    def test_demix_3d_output_handling(self):
        """Test that demix handles 3D model outputs correctly without dimension mismatches."""
        class Mock3DModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                # Returns (batch, channels, samples) - 3D shape like Apollo
                return torch.zeros_like(x)

        config = ConfigDict({
            'audio': {'chunk_size': 4000, 'sample_rate': 44100, 'num_channels': 2},
            'training': {'instruments': ['restored', 'addition'], 'target_instrument': 'restored', 'model_type': 'apollo'},
            'inference': {'batch_size': 1, 'num_overlap': 2}
        })

        model = Mock3DModel()
        mix = np.zeros((2, 8000), dtype=np.float32)
        res = demix(config, model, mix, torch.device('cpu'), 'apollo', pbar=False)
        self.assertIn('restored', res)
        self.assertEqual(res['restored'].shape, (2, 8000))

    def test_feature_dim_sniffing_from_checkpoint(self):
        """Test that get_model_from_config sniffs feature_dim from checkpoint when config differs."""
        import tempfile
        # Create a mock state dict with BN.0.1.weight shape (384, ...)
        mock_ckpt = {
            'state_dict': {
                'audio_model.BN.0.1.weight': torch.zeros((384, 11, 1))
            }
        }
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
            tmp_ckpt = f.name
            torch.save(mock_ckpt, tmp_ckpt)
        try:
            # config_apollo.yaml specifies feature_dim: 256
            model, config = get_model_from_config(
                'apollo', 'configs/config_apollo.yaml', checkpoint_path=tmp_ckpt
            )
            self.assertEqual(model.feature_dim, 384)
        finally:
            if os.path.exists(tmp_ckpt):
                os.remove(tmp_ckpt)


if __name__ == '__main__':
    unittest.main()
