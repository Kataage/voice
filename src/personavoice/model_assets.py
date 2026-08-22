from __future__ import annotations

# Upstream source/model revisions are intentionally centralized here so setup,
# preparation, training, inference, and doctor all agree on the exact assets
# PersonaVoice was audited against.

IRODORI_SOURCE_REVISION = "8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
SEED_VC_SOURCE_REVISION = "51383efd921027683c89e5348211d93ff12ac2a8"

IRODORI_MODEL_ID = "Aratako/Irodori-TTS-v4.1-Small"
IRODORI_MODEL_FILENAME = "model.safetensors"
# v4.1 is a separate Hugging Face repository from v4-Small. The old c0097bd...
# revision belongs to Aratako/Irodori-TTS-v4-Small and therefore 404s when used
# against the v4.1 repository. df20e70... is the audited v4.1 upload commit that
# contains the checkpoint with the SHA256 pinned below.
IRODORI_MODEL_REVISION = "df20e70ca3a5ceaedda6cb98e82029b68c800857"
IRODORI_MODEL_SHA256 = "c85de88c01700cb53538e706f128ebcb1b8513ad21d7d0e75f58bc82cdbf89f6"

IRODORI_DACVAE_ID = "Aratako/Semantic-DACVAE-Japanese-32dim"
IRODORI_DACVAE_FILENAME = "weights.pth"
IRODORI_DACVAE_REVISION = "818b64119d4fdc2476ccdce7836d746b279ae3fa"
IRODORI_DACVAE_SHA256 = "db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5"

# This exact revision is referenced by the pinned Irodori v4 Small training
# configs. Materializing a different ModernBERT revision can break offline
# training even when the TTS checkpoint itself is unchanged.
IRODORI_TEXT_ENCODER_ID = "sbintuitions/modernbert-ja-310m"
IRODORI_TEXT_ENCODER_REVISION = "77675fc96a7e445e982e2ba90246b816efc74ec6"

LFM_MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
LFM_MODEL_REVISION = "b31023f2d69b95fbd7876898f8de9fae90e8afbd"
LFM_MODEL_WEIGHT_SHA256 = "abf38960d3f37c2be7c946a9b6b06d23ed04a1afb8ac192aa3b491e3dcdcf325"

ASR_MODEL_ID = "Systran/faster-whisper-large-v3"
ASR_MODEL_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
ASR_MODEL_WEIGHT_SHA256 = "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1"

PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"
PYANNOTE_MODEL_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
PYANNOTE_MODEL_ASSET_SHA256 = {
    "config.yaml": "5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be",
    "embedding/pytorch_model.bin": "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929",
    "segmentation/pytorch_model.bin": "7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e",
    "plda/plda.npz": "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255",
    "plda/xvec_transform.npz": "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f",
}

# SenseVoice is distributed through ModelScope without a stable commit contract
# exposed by our current downloader. Pin the inference-critical assets instead.
# The worker verifies these hashes before trusting a downloaded/local model and
# uses the uv-locked FunASR implementation (trust_remote_code=False).
SENSE_MODEL_ID = "iic/SenseVoiceSmall"
SENSE_MODEL_WEIGHT_SHA256 = "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"
SENSE_MODEL_CMVN_SHA256 = "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5"
SENSE_MODEL_TOKENIZER_SHA256 = "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8"
