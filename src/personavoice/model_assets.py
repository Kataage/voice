from __future__ import annotations

# Upstream source/model revisions are intentionally centralized here so setup,
# preparation, training, inference, and doctor all agree on the exact assets
# PersonaVoice was audited against.

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

PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"
PYANNOTE_MODEL_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"

# SenseVoice is distributed through ModelScope without a stable commit contract
# exposed by our current downloader. Pin the inference-critical assets instead.
# The worker verifies these hashes before trusting a downloaded/local model and
# uses the uv-locked FunASR implementation (trust_remote_code=False).
SENSE_MODEL_ID = "iic/SenseVoiceSmall"
SENSE_MODEL_WEIGHT_SHA256 = "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"
SENSE_MODEL_CMVN_SHA256 = "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5"
SENSE_MODEL_TOKENIZER_SHA256 = "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8"
