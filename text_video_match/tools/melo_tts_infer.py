#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeloTTS 批量推理脚本。
在独立环境中一次加载模型，逐句输出音频文件。
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nltk
import unidic_lite
from melo.api import TTS


def ensure_nltk_resource(name: str, path: str) -> None:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, quiet=True)


def get_audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MeloTTS batch inference helper")
    parser.add_argument("--input-json", required=True, help="Batch input json path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--language", default="ZH", help="Melo language code")
    parser.add_argument("--speaker", default="ZH", help="Speaker key in spk2id")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    ensure_nltk_resource("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng")
    ensure_nltk_resource("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger")
    ensure_nltk_resource("cmudict", "corpora/cmudict")

    os.environ["MECABRC"] = os.path.join(unidic_lite.DICDIR, "mecabrc")

    input_path = Path(args.input_json)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    requests = payload.get("requests", [])

    print(f"[melo] requests={len(requests)} device={args.device} language={args.language} speaker={args.speaker}", flush=True)
    print("[melo] loading model...", flush=True)
    model = TTS(language=args.language, device=args.device)
    print("[melo] model loaded", flush=True)
    speaker_ids = model.hps.data.spk2id
    if args.speaker not in speaker_ids:
        raise KeyError(f"speaker not found: {args.speaker}, available={list(speaker_ids.keys())}")

    results = []
    total = len(requests)
    for idx, item in enumerate(requests, 1):
        item_id = item["id"]
        text = item["text"]
        output_path = Path(item["output"])
        preview = text[:30].replace("\n", " ")
        print(f"[melo] {idx}/{total} start id={item_id} text={preview}", flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        with tempfile.TemporaryDirectory(prefix=f"melo_tts_{item_id}_") as tmp_dir:
            tmp_wav = Path(tmp_dir) / "melo.wav"
            model.tts_to_file(
                text,
                speaker_ids[args.speaker],
                str(tmp_wav),
                speed=args.speed,
            )
            print(f"[melo] {idx}/{total} wav_done id={item_id}", flush=True)

            if output_path.suffix.lower() == ".wav":
                tmp_wav.replace(output_path)
            else:
                print(f"[melo] {idx}/{total} transcode_start id={item_id}", flush=True)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(tmp_wav),
                        "-codec:a",
                        "libmp3lame",
                        "-qscale:a",
                        "2",
                        str(output_path),
                    ],
                    capture_output=True,
                    check=True,
                )
                print(f"[melo] {idx}/{total} transcode_done id={item_id}", flush=True)

        duration = get_audio_duration(output_path)
        results.append(
            {
                "id": item_id,
                "output": str(output_path),
                "duration": duration,
            }
        )
        print(f"[melo] {idx}/{total} done id={item_id} duration={duration:.3f}", flush=True)

    print(json.dumps({"results": results}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
