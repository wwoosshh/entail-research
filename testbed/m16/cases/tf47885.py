"""M16 case, huggingface/transformers#47885 (testbed/M16_PROTOCOL.md 5): one non-finite sample in a 16,000-sample
waveform makes WhisperFeatureExtractor return an all-NaN feature matrix (the dynamic-range floor uses max()), and
the ASR pipeline transcribes it as '0' with no error. The report's own script (openai/whisper-tiny, CPU), entail off
or on from outside; a clean copy of the waveform is transcribed as a control.
Run in ~/venvs/gpu: python testbed/m16/cases/tf47885.py <out.json>
"""
import json
import os
import sys


def main():
    import numpy as np
    from transformers import WhisperFeatureExtractor, pipeline

    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000) * 0.05).astype(np.float32)
    clean = audio.copy()
    audio[8000] = np.nan
    feats = fe(audio, sampling_rate=16000, return_tensors="np")["input_features"]
    nan_fraction = float(np.isnan(feats).mean())
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu")
    err = None
    text_bad = None
    try:
        text_bad = asr(audio)["text"]
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:300]
    text_clean = asr(clean)["text"]
    row = {"entail": os.environ.get("ENTAIL", "off"), "non_finite_input_samples": 1, "input_samples": int(audio.size),
           "feature_nan_fraction": nan_fraction, "transcription_with_nan": text_bad, "error": err,
           "transcription_clean": text_clean,
           "reproduced": nan_fraction == 1.0 and err is None and isinstance(text_bad, str)}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
