#!/usr/bin/env python3

import argparse
import io
import os
import tempfile
import wave

import numpy as np
from faster_whisper import WhisperModel


# =====================================================
# 모드별 initial_prompt 정의
# =====================================================
PROMPTS = {
    "restaurant": (
        "A customer is ordering food or drinks at a restaurant."
    ),
    "receptionist": (
        "A guest is introducing themselves to a receptionist. "
        "The guest states their name and favorite drink."
    ),
    "gpsr": (
        "A user is giving a task instruction to a home service robot. "
        "The instruction describes an action to perform at a specific location."
    ),
}

DEFAULT_PROMPT = "A person is speaking a command or request in English."


def get_prompt(mode: str) -> str:
    """mode 문자열에 따라 initial_prompt 반환"""
    if not mode:
        return DEFAULT_PROMPT

    return PROMPTS.get(mode.strip().lower(), DEFAULT_PROMPT)


def amplify_wav_file(input_path: str, output_path: str, gain_db: float) -> None:
    """
    WAV 파일의 음량을 증폭해 새로운 WAV 파일로 저장.
    현재 코드와 동일하게 16-bit PCM WAV를 기준으로 처리.
    """
    with wave.open(input_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(
            f"Expected 16-bit PCM WAV file, but got sample width={sampwidth}"
        )

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)

    gain = 10 ** (gain_db / 20.0)
    audio *= gain

    audio = np.clip(audio, -32768, 32767).astype(np.int16)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(audio.tobytes())


class WhisperWavTranscriber:
    def __init__(self, model_path: str):
        self.model_path = model_path

        print(f"[INFO] Loading faster-whisper model: {self.model_path}")

        try:
            self.model = WhisperModel(
                self.model_path,
                device="cuda",
                compute_type="float16",
                local_files_only=True
            )
            print("[INFO] Model loaded on CUDA (float16).")

        except Exception as e:
            print(f"[WARN] CUDA load failed: {e}")
            print("[INFO] Falling back to CPU (int8).")

            self.model = WhisperModel(
                self.model_path,
                device="cpu",
                compute_type="int8",
                local_files_only=True
            )

    def transcribe(
        self,
        wav_path: str,
        mode: str = "restaurant",
        gain_db: float = 20.0
    ) -> str:
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        prompt = get_prompt(mode)

        print(f"[INFO] Input WAV: {wav_path}")
        print(f"[INFO] Mode: {mode}")
        print(f"[INFO] Gain: {gain_db} dB")
        print(f"[INFO] Prompt: {prompt}")

        temp_path = None

        try:
            # 증폭된 임시 WAV 파일 생성
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as temp_file:
                temp_path = temp_file.name

            amplify_wav_file(
                input_path=wav_path,
                output_path=temp_path,
                gain_db=gain_db
            )

            # Whisper 변환
            segments, info = self.model.transcribe(
                temp_path,
                language="en",
                beam_size=5,
                initial_prompt=prompt,
                condition_on_previous_text=False,
                temperature=0.0,
                vad_filter=True,
                no_repeat_ngram_size=6
            )

            text = " ".join(segment.text for segment in segments).strip()

            print(f"[INFO] Detected language: {info.language}")
            return text

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a WAV file using faster-whisper."
    )

    parser.add_argument(
        "wav_path",
        type=str,
        help="Path to input WAV file"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="restaurant",
        choices=["restaurant", "receptionist", "gpsr"],
        help="Prompt mode for transcription"
    )

    parser.add_argument(
        "--gain-db",
        type=float,
        default=20.0,
        help="Audio amplification gain in dB. Default: 20.0"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get(
            "WHISPER_MODEL_PATH",
            "/root/.cache/faster-whisper/large-v3-turbo"
        ),
        help="Path to local faster-whisper model"
    )

    args = parser.parse_args()

    transcriber = WhisperWavTranscriber(args.model_path)

    try:
        text = transcriber.transcribe(
            wav_path=args.wav_path,
            mode=args.mode,
            gain_db=args.gain_db
        )

        if text:
            print("\n========== RESULT ==========")
            print(text)
            print("============================")
        else:
            print("\n[INFO] No speech detected.")

    except Exception as e:
        print(f"[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()
