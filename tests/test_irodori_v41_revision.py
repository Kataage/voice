from personavoice import model_assets

V4_MODEL_REVISION = "c0097bd1be75833b44498bb6fcf0bee9298262a7"
V41_MODEL_REVISION = "df20e70ca3a5ceaedda6cb98e82029b68c800857"
V41_MODEL_SHA256 = "c85de88c01700cb53538e706f128ebcb1b8513ad21d7d0e75f58bc82cdbf89f6"


def test_irodori_v41_uses_its_own_huggingface_revision() -> None:
    assert model_assets.IRODORI_MODEL_ID == "Aratako/Irodori-TTS-v4.1-Small"
    assert model_assets.IRODORI_MODEL_REVISION == V41_MODEL_REVISION
    assert model_assets.IRODORI_MODEL_REVISION != V4_MODEL_REVISION
    assert model_assets.IRODORI_MODEL_SHA256 == V41_MODEL_SHA256
