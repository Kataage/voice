from __future__ import annotations

# Upstream source/model revisions are intentionally centralized here so setup,
# preparation, training, inference, and doctor all agree on the exact assets
# PersonaVoice was audited against.

IRODORI_SOURCE_REVISION = "8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
SEED_VC_SOURCE_REVISION = "51383efd921027683c89e5348211d93ff12ac2a8"

# Vevo2's public FM-only inference path is currently maintained in Amphion.
# These values are copied from the audited upstream/Hugging Face revisions; do
# not replace them with a floating branch or an unverified mirror.
VEVO2_SOURCE_REPOSITORY = "https://github.com/open-mmlab/Amphion.git"
VEVO2_SOURCE_REVISION = "26f6883110181f1dbfe95c70a7c7dbaf4de5f42a"
VEVO2_SOURCE_LICENSE = "MIT"
VEVO2_SOURCE_LICENSE_URL = "https://github.com/open-mmlab/Amphion/blob/main/LICENSE"
VEVO2_MODEL_ID = "RMSnow/Vevo2"
VEVO2_MODEL_REVISION = "2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8"
VEVO2_MODEL_LICENSE = "CC BY-NC-ND 4.0"
VEVO2_MODEL_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-nd/4.0/"
VEVO2_MODEL_ASSET_SHA256 = {
    "acoustic_modeling/fm_emilia101k_singnet7k_repa/config.json":
        "e0ce1432ac20903ec8c65d16794e93c278a11d0a472ab0032241c20b8d07a4db",
    "acoustic_modeling/fm_emilia101k_singnet7k_repa/model.safetensors":
        "ef3733b3f92cf8f38e32f6d161f58751247b199cffc15cc1562e76c1289a7186",
    "acoustic_modeling/fm_emilia101k_singnet7k_repa/whisper_stats.pt":
        "6117052b3e23e9075a79cc208ecc24328ae2f71a7dd4c9793db7c90a88b4a519",
    "tokenizer/contentstyle_fvq16384_12.5hz/model.safetensors":
        "bebdcea39e2d0134dbcce193aed0e7b0d393a866346317cd196389e40e9151b0",
    "vocoder/config.json":
        "a6445bcc2182c43d97491b38d79022901f02b0f9fb56aa7a51498d0da8b9767d",
    "vocoder/model.safetensors":
        "5b5d1a46b19351c9a71bd8a5a59dd16be0be2ddefe70d3a0b4915d9a425e56d3",
    "vocoder/model_1.safetensors":
        "850799d78699134b969056183fc9d490c51f8d8154d0ed00fae3e738b6b30af6",
    "vocoder/model_2.safetensors":
        "56130fd13d5fbe828e56d61edb0049d35700db0472a866b8167d1d217d2687f8",
}
VEVO2_MODEL_REQUIRED_FILES = tuple(sorted(VEVO2_MODEL_ASSET_SHA256))
VEVO2_WHISPER_MODEL_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"
)
VEVO2_WHISPER_MODEL_SHA256 = "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1"
VEVO2_WHISPER_SOURCE_REVISION = "5f86d1d86363843179951550570367b37c5d6f78"
VEVO2_WHISPER_LICENSE = "MIT"
VEVO2_WHISPER_LICENSE_URL = "https://github.com/openai/whisper/blob/main/LICENSE"

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
# Complete Transformers architecture/tokenizer contract downloaded from the
# exact revision above.  The non-weight files influence module construction,
# token IDs, and chat-template rendering just as directly as the weights do;
# binding only model.safetensors would allow local and Modal training to run
# different algorithms under one family fingerprint.
LFM_MODEL_ASSET_SHA256 = {
    "chat_template.jinja": "89e790f027916b5a2bca145a6a8454e06ffc7a5043bf3b6d97829aff86bb543f",
    "config.json": "df8dac1ebef28c06a010be6353e7dd2d0a3ff9c2ca23591bb8ced252d74510a1",
    "model.safetensors": LFM_MODEL_WEIGHT_SHA256,
    "special_tokens_map.json": "742aefe2b7dec496e8caffdba03a75d0c1a9925d53bd3f3e0d388c96b591b6f4",
    "tokenizer.json": "d7a0ab0fc22e41ec8c6d7450a9ff9ce40e196ec5e5a2fa6a2105e064e0514ed7",
    "tokenizer_config.json": "8cba5b0c7acab23a0d4cc9ac587346c9220a1b6d288fc5346fe118202fd6f43e",
}
LFM_MODEL_REQUIRED_FILES = tuple(sorted(LFM_MODEL_ASSET_SHA256))

ASR_MODEL_ID = "Systran/faster-whisper-large-v3"
ASR_MODEL_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
ASR_MODEL_WEIGHT_SHA256 = "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1"

# Modern ASR assets are kept next to the legacy Whisper pins so every layer
# (config, worker, Prepare lineage, setup and doctor) can consume one audited
# registry.  Hugging Face LFS object IDs are recorded as integrity evidence;
# they are not mislabeled as SHA256 checksums of the local files.
QWEN_ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN_ASR_MODEL_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
QWEN_ASR_MODEL_LICENSE = "Apache-2.0"
QWEN_ASR_MODEL_REQUIRED_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
)
QWEN_ASR_MODEL_LFS_OIDS = {
    "model-00001-of-00002.safetensors":
        "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6",
    "model-00002-of-00002.safetensors":
        "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc",
}

QWEN_DOMAIN_ASR_MODEL_ID = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
QWEN_DOMAIN_ASR_MODEL_REVISION = "5a6a789ceb2f22d2b8606743b13a8159af218362"
QWEN_DOMAIN_ASR_MODEL_PAGE_LICENSE = "Apache-2.0"
QWEN_DOMAIN_ASR_DATASET_ID = "litagin/Galgame_Speech_ASR_16kHz"
QWEN_DOMAIN_ASR_DATASET_REVISION = "3fb86654222b3f0af0f7c332ae6a0ef9752a9451"
QWEN_DOMAIN_ASR_DATASET_LICENSE = "GPL-3.0"
QWEN_DOMAIN_ASR_EFFECTIVE_RESTRICTIONS = (
    "commercial-use-prohibition",
    "models-trained-on-dataset-must-be-open-sourced",
)
QWEN_DOMAIN_ASR_MODEL_REQUIRED_FILES = (
    "chat_template.jinja",
    "config.json",
    "conversion_report.json",
    "generation_config.json",
    "model-00001-of-00006.safetensors",
    "model-00002-of-00006.safetensors",
    "model-00003-of-00006.safetensors",
    "model-00004-of-00006.safetensors",
    "model-00005-of-00006.safetensors",
    "model-00006-of-00006.safetensors",
    "model.safetensors.index.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "ctc_aligner.pt",
)
QWEN_DOMAIN_ASR_CTC_HEAD_FILE = "ctc_aligner.pt"
QWEN_DOMAIN_ASR_CTC_HEAD_LFS_OID = (
    "61ef0ccd0e18f26adbf3bccc58165e4534bf727a793f02b363d24c556257b911"
)

QWEN_FORCED_ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
QWEN_FORCED_ALIGNER_MODEL_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
QWEN_FORCED_ALIGNER_MODEL_LICENSE = "Apache-2.0"
QWEN_FORCED_ALIGNER_MODEL_REQUIRED_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
)
QWEN_FORCED_ALIGNER_MODEL_LFS_OIDS = {
    "model.safetensors":
        "47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7",
}

# BGM-aware analysis uses the actively maintained audio-separator wrapper at a
# fixed release.  The UVR model is downloaded/materialized separately and is
# never bundled or redistributed by PersonaVoice.  Its local manifest records
# the actual model digest before an offline run is admitted.
SEPARATOR_PACKAGE = "audio-separator"
SEPARATOR_VERSION = "0.44.2"
SEPARATOR_SOURCE_REPOSITORY = "https://github.com/nomadkaraoke/python-audio-separator"
SEPARATOR_SOURCE_REVISION = "fca0cf76d52b545cedbc75e1d3aea626d513c036"
SEPARATOR_SOURCE_LICENSE = "MIT"
SEPARATOR_MODEL_FILENAME = "UVR_MDXNET_KARA_2.onnx"
SEPARATOR_MODEL_LINEAGE = "UVR-MDXNET-KARA-2"
SEPARATOR_MODEL_LICENSE = "UVR model-specific terms; local materialization only"

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
