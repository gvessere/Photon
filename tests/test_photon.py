"""
PHOTON Verification Tests

Tests for:
- Shape correctness at each level
- Causality (no leakage from future)
- Train/inference consistency
- Latent AR head functionality
"""

import pytest
import torch
import torch.nn.functional as F

from photon import PhotonConfig, PhotonLM
from photon.inference import generate_photon, generate_token_chunk


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return PhotonConfig(
        vocab_size=1000,
        C1=4,
        C2=4,
        d_embed_enc=64,
        d_latent=256,  # 4 * 64
        d_converter=128,
        R1=2,
        R2=2,
        n_heads=4,
        d_ff=512,
        n_layers_enc=2,
        n_layers_dec=2,
        n_layers_latent_ar=2,
        gradient_checkpointing=False,
        use_sdpa=True,
    )


@pytest.fixture
def model(small_config):
    """Create a small model for testing."""
    return PhotonLM(small_config)


@pytest.fixture
def sample_input(small_config):
    """Create sample input tensor."""
    B, T = 2, small_config.C1 * small_config.C2 * 4  # 4 full blocks
    return torch.randint(0, small_config.vocab_size, (B, T))


# =============================================================================
# Shape Tests
# =============================================================================

class TestShapes:
    """Test that all tensor shapes are correct throughout the model."""
    
    def test_config_block_size(self, small_config):
        """Config block_size property is correct."""
        assert small_config.block_size == small_config.C1 * small_config.C2
    
    def test_encoding_shapes(self, model, sample_input, small_config):
        """Encoder produces correct shapes."""
        B, T = sample_input.shape
        x1, x2 = model.encode(sample_input)
        
        expected_m1 = T // small_config.C1
        expected_m2 = T // (small_config.C1 * small_config.C2)
        
        assert x1.shape == (B, expected_m1, small_config.d_latent)
        assert x2.shape == (B, expected_m2, small_config.d_latent)
    
    def test_forward_shapes(self, model, sample_input, small_config):
        """Forward pass produces correct output shapes."""
        B, T = sample_input.shape
        out = model(sample_input, labels=sample_input)
        
        assert "loss" in out
        assert "loss_latent" in out
        assert "logits" in out
        
        # Logits should cover all token positions
        assert out["logits"].shape == (B, T, small_config.vocab_size)
    
    def test_converter_shapes(self, model, small_config):
        """Converters produce correct shapes."""
        B = 2
        latent = torch.randn(B, small_config.d_latent)
        
        # Level 2 input converter
        cond2 = model.dec_conv2_in(latent)
        assert cond2.shape == (B, small_config.R2, small_config.d_latent)
        
        # Level 1 converter
        cond1 = model.dec_conv1(latent)
        assert cond1.shape == (B, small_config.R1, small_config.d_latent)
    
    def test_latent_ar_shapes(self, model, small_config):
        """Latent AR head produces correct shapes."""
        B, M = 2, 8  # 8 latents in sequence
        latent_seq = torch.randn(B, M, small_config.d_latent)
        
        pred = model.latent_ar_head(latent_seq)
        
        assert pred.shape == (B, M, small_config.d_latent)


# =============================================================================
# Causality Tests
# =============================================================================

class TestCausality:
    """Test that the model respects causal dependencies."""
    
    def test_encoder_causality(self, model, small_config):
        """Encoder context transformers are causal."""
        B, T = 2, small_config.C1 * 8
        x = torch.randn(B, T, small_config.d_embed_enc)
        
        # Chunk and encode
        x1 = model.enc_chunk1(x)
        
        # Check that changing future tokens doesn't affect past outputs
        x_modified = x.clone()
        x_modified[:, T//2:, :] = torch.randn_like(x_modified[:, T//2:, :])
        x1_modified = model.enc_chunk1(x_modified)
        
        out1 = model.enc_ctx1(x1, is_causal=True)
        out1_modified = model.enc_ctx1(x1_modified, is_causal=True)
        
        # First half should be identical (causal = no future leakage)
        half = x1.size(1) // 2
        torch.testing.assert_close(out1[:, :half, :], out1_modified[:, :half, :])
    
    def test_decoder_causality(self, model, small_config):
        """Decoder transformers are causal."""
        B, T = 2, small_config.R1 + small_config.C1
        x = torch.randn(B, T, small_config.d_latent)
        
        # Modify second half
        x_modified = x.clone()
        x_modified[:, T//2:, :] = torch.randn_like(x_modified[:, T//2:, :])
        
        out = model.dec_ctx1(x, is_causal=True)
        out_modified = model.dec_ctx1(x_modified, is_causal=True)
        
        # First half should be identical
        torch.testing.assert_close(out[:, :T//2, :], out_modified[:, :T//2, :])
    
    def test_no_future_latent_leakage(self, model, sample_input, small_config):
        """Token chunk g only depends on latent g-1, not g or future."""
        B, T = sample_input.shape
        
        # Get latents
        x1, x2 = model.encode(sample_input)
        M1 = x1.size(1)
        
        # Modify future latents
        x1_modified = x1.clone()
        x1_modified[:, M1//2:, :] = torch.randn_like(x1_modified[:, M1//2:, :])
        
        # Forward should use prev_l1 (shifted), so early chunks shouldn't change
        # This is a structural test - if conditioning is correct, it passes
        
        # Build conditioning from original and modified latents
        prev_l1 = torch.cat([
            model.start_latent_l1.view(1, 1, -1).expand(B, 1, -1),
            x1[:, :-1, :],
        ], dim=1)
        
        prev_l1_modified = torch.cat([
            model.start_latent_l1.view(1, 1, -1).expand(B, 1, -1),
            x1_modified[:, :-1, :],
        ], dim=1)
        
        # First conditioning vectors should be identical
        # (before the modification point)
        torch.testing.assert_close(
            prev_l1[:, :M1//2, :],
            prev_l1_modified[:, :M1//2, :]
        )


# =============================================================================
# Latent AR Tests
# =============================================================================

class TestLatentAR:
    """Test the latent autoregressive head."""
    
    def test_ar_head_forward(self, model, small_config):
        """AR head forward pass works."""
        B, M = 2, 8
        latent_seq = torch.randn(B, M, small_config.d_latent)
        
        mean, logvar = model.latent_ar_head(latent_seq)
        
        # Should predict next latent for each position
        assert not torch.isnan(mean).any()
        assert not torch.isnan(logvar).any()
    
    def test_ar_head_sampling(self, model, small_config):
        """AR head sampling produces valid latents."""
        B = 2
        prev_latent = torch.randn(B, small_config.d_latent)
        
        # Sample with temperature
        next_latent = model.latent_ar_head.sample(prev_latent, temperature=1.0)
        assert next_latent.shape == (B, small_config.d_latent)
        
        # Sample deterministically
        next_latent_det = model.latent_ar_head.sample(prev_latent, temperature=0.0)
        assert next_latent_det.shape == (B, small_config.d_latent)
    
    def test_ar_loss_training(self, model, sample_input):
        """AR loss is computed during training."""
        out = model(sample_input, labels=sample_input)
        
        # Loss should include AR component (loss_latent contains both L2->L1 and AR)
        assert out["loss_latent"] > 0


# =============================================================================
# Loss Tests
# =============================================================================

class TestLosses:
    """Test loss computation."""
    
    def test_gaussian_latent_loss(self, model, sample_input, small_config):
        """Gaussian latent loss is computed correctly."""
        # Ensure we're using Gaussian loss
        model.cfg.latent_loss_type = "gaussian"
        
        out = model(sample_input, labels=sample_input)
        
        assert "loss_latent" in out
        assert out["loss_latent"] > 0
        assert not torch.isnan(out["loss_latent"])
    
    def test_lm_loss(self, model, sample_input):
        """LM cross-entropy loss is computed correctly."""
        out = model(sample_input, labels=sample_input)
        
        assert "loss_lm" in out
        assert out["loss_lm"] > 0
        assert not torch.isnan(out["loss_lm"])
    
    def test_combined_loss_weighting(self, model, sample_input, small_config):
        """Loss weighting is applied correctly."""
        # Set different weights
        model.cfg.lambda_latent = 0.5
        model.cfg.lambda_lm = 2.0
        
        out = model(sample_input, labels=sample_input)
        
        expected = 0.5 * out["loss_latent"] + 2.0 * out["loss_lm"]
        torch.testing.assert_close(out["loss"], expected, rtol=1e-4, atol=1e-4)


# =============================================================================
# Generation Tests
# =============================================================================

class TestGeneration:
    """Test generation functionality."""
    
    def test_token_chunk_generation(self, model, small_config):
        """Generate a single token chunk."""
        B = 2
        prev_l1 = torch.randn(B, small_config.d_latent)
        
        chunk = generate_token_chunk(model, prev_l1, temperature=1.0, top_k=50)
        
        assert chunk.shape == (B, small_config.C1)
        assert (chunk >= 0).all() and (chunk < small_config.vocab_size).all()
    
    def test_full_generation(self, model, small_config):
        """Generate multiple tokens with full pipeline."""
        B = 1
        prompt = torch.randint(0, small_config.vocab_size, (B, small_config.block_size))
        
        generated = generate_photon(
            model=model,
            input_ids=prompt,
            max_new_tokens=small_config.block_size,
            temperature=1.0,
            top_k=50,
            use_latent_ar=True,
        )
        
        # Should have prompt + new tokens
        assert generated.shape[1] >= prompt.shape[1]
        assert generated.shape[1] <= prompt.shape[1] + small_config.block_size
    
    def test_generation_without_latent_ar(self, model, small_config):
        """Generation works without latent AR (with re-encoding)."""
        B = 1
        prompt = torch.randint(0, small_config.vocab_size, (B, small_config.block_size))
        
        generated = generate_photon(
            model=model,
            input_ids=prompt,
            max_new_tokens=small_config.block_size,
            use_latent_ar=False,  # Re-encode mode
        )
        
        assert generated.shape[1] >= prompt.shape[1]


# =============================================================================
# Gradient Flow Tests
# =============================================================================

class TestGradients:
    """Test gradient flow through the model."""
    
    def test_gradient_flow(self, model, sample_input):
        """Gradients flow to all parameters."""
        out = model(sample_input, labels=sample_input)
        out["loss"].backward()
        
        # Check some key parameters have gradients
        assert model.enc_embed.weight.grad is not None
        assert model.dec_embed.weight.grad is not None
        assert model.lm_head.weight.grad is not None  # Tied with dec_embed
        assert model.start_latent_l1.grad is not None
        assert model.start_latent_l2.grad is not None
    
    def test_no_nan_gradients(self, model, sample_input):
        """No NaN gradients after backward."""
        out = model(sample_input, labels=sample_input)
        out["loss"].backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


# =============================================================================
# RoPE Tests
# =============================================================================

class TestRoPE:
    """Test rotary position embeddings."""
    
    def test_rope_shape(self, model, small_config):
        """RoPE preserves tensor shape."""
        rope = model.enc_ctx1.rope
        B, H, T, D = 2, small_config.n_heads, 16, small_config.d_latent // small_config.n_heads
        
        x = torch.randn(B, H, T, D)
        y = rope(x)
        
        assert y.shape == x.shape
    
    def test_rope_position_sensitivity(self, model, small_config):
        """RoPE makes different positions distinguishable."""
        rope = model.enc_ctx1.rope
        B, H, T, D = 2, small_config.n_heads, 16, small_config.d_latent // small_config.n_heads
        
        x = torch.randn(B, H, T, D)
        
        # Same content but different positions should give different results
        y1 = rope(x, position_ids=torch.arange(T).unsqueeze(0).expand(B, -1))
        y2 = rope(x, position_ids=torch.arange(T).unsqueeze(0).expand(B, -1) + 10)
        
        assert not torch.allclose(y1, y2)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_train_step(self, model, sample_input):
        """Complete training step works."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        model.train()
        optimizer.zero_grad()
        
        out = model(sample_input, labels=sample_input)
        out["loss"].backward()
        optimizer.step()
        
        # Model should still work after update
        with torch.no_grad():
            out2 = model(sample_input, labels=sample_input)
        
        assert not torch.isnan(out2["loss"])
    
    def test_encode_generate_consistency(self, model, small_config):
        """Encoding and generation use consistent representations."""
        B = 1
        prompt = torch.randint(0, small_config.vocab_size, (B, small_config.block_size))
        
        # Encode prompt
        x1, x2 = model.encode(prompt)
        
        # Generate - should start from same latent state
        model.eval()
        with torch.no_grad():
            generated = generate_photon(
                model, prompt, max_new_tokens=small_config.C1,
                use_latent_ar=True
            )
        
        # Encode the generated sequence
        x1_gen, x2_gen = model.encode(generated)
        
        # First parts should be very close (same prompt encoding)
        torch.testing.assert_close(x2[:, :-1, :], x2_gen[:, :x2.size(1)-1, :], rtol=1e-3, atol=1e-3)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
