"""TTS voice selection follows config.yaml (tts.voice / tts.lang_code)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from jarvis import config, speaker

FRENCH = {"voice": "ff_siwis", "lang_code": "f"}


class ResolveVoiceTest(unittest.TestCase):
    def test_voice_and_lang_code_come_from_tts_config(self):
        self.assertEqual(config.resolve_voice(FRENCH), ("ff_siwis", "f"))

    def test_defaults_when_tts_config_is_empty(self):
        self.assertEqual(config.resolve_voice({}), ("af_heart", "a"))

    def test_per_language_entry_overrides_configured_voice(self):
        tts = {**FRENCH, "voices": {"it": {"voice": "if_sara", "lang_code": "i"}}}
        self.assertEqual(config.resolve_voice(tts, "it"), ("if_sara", "i"))

    def test_language_without_entry_keeps_configured_voice(self):
        tts = {**FRENCH, "voices": {"it": {"voice": "if_sara", "lang_code": "i"}}}
        self.assertEqual(config.resolve_voice(tts, "en"), ("ff_siwis", "f"))


class _RecordingModel:
    """Stands in for Kokoro: records what render() asks it to generate."""

    def __init__(self):
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append(kwargs)
        yield SimpleNamespace(audio=np.zeros(240, dtype=np.float32))


class RenderVoiceTest(unittest.TestCase):
    def test_render_with_text_only_uses_configured_voice(self):
        model = _RecordingModel()
        cfg = {"tts": {**FRENCH, "speed": 1.0}}
        with patch.object(speaker, "_get_model", return_value=model), \
                patch.object(config, "get_config", return_value=cfg):
            speaker.render("Bonjour")

        self.assertEqual(model.calls[0]["voice"], "ff_siwis")
        self.assertEqual(model.calls[0]["lang_code"], "f")


if __name__ == "__main__":
    unittest.main()
