from __future__ import annotations

# Upstream source/model revisions are intentionally centralized here so setup,
# training, inference, and doctor all agree on the exact assets PersonaVoice was
# audited against.

IRODORI_MODEL_ID = "Aratako/Irodori-TTS-v4.1-Small"
IRODORI_MODEL_FILENAME = "model.safetensors"
IRODORI_MODEL_SHA256 = "c85de88c01700cb53538e706f128ebcb1b8513ad21d7d0e75f58bc82cdbf89f6"

IRODORI_DACVAE_ID = "Aratako/Semantic-DACVAE-Japanese-32dim"
IRODORI_DACVAE_FILENAME = "weights.pth"
IRODORI_DACVAE_SHA256 = "db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5"

# This exact revision is referenced by the pinned Irodori v4 Small training
# configs. Materializing a different ModernBERT revision can break offline
# training even when the TTS checkpoint itself is unchanged.
IRODORI_TEXT_ENCODER_ID = "sbintuitions/modernbert-ja-310m"
IRODORI_TEXT_ENCODER_REVISION = "77675fc96a7e445e982e2ba90246b816efc74ec6"

LFM_MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
LFM_MODEL_REVISION = "b31023f2d69b95fbd7876898f8de9fae90e8afbd"

ASR_MODEL_ID = "Systran/faster-whisper-large-v3"
ASR_MODEL_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
