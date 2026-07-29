#!/usr/bin/env python3
"""WER validation against LibriSpeech test-clean (or any split with the same
{speaker}/{chapter}/*.trans.txt + *.flac layout).

Uses openai-whisper's own EnglishTextNormalizer before scoring — the same
normalization OpenAI's Whisper paper applies before computing WER (strips
casing/punctuation differences that would otherwise inflate the error count
without reflecting real transcription mistakes).

Usage (inside dev container):
    python scripts/validate_librispeech.py --model base.en --limit 100
    python scripts/validate_librispeech.py --model base.en --backend whisper
"""

import argparse
import json
import time
from pathlib import Path

import jiwer
from whisper.normalizers import EnglishTextNormalizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/dataset/librispeech/LibriSpeech/test-clean")
    p.add_argument("--model", choices=["tiny.en", "base.en", "small.en"], default="base.en")
    p.add_argument("--backend", choices=["whisper_trt", "whisper"], default="whisper_trt")
    p.add_argument("--limit", type=int, default=None, help="Only score the first N utterances")
    p.add_argument("--results-json", default=None)
    return p.parse_args()


def iter_utterances(root: Path):
    for trans_file in sorted(root.rglob("*.trans.txt")):
        for line in trans_file.read_text().splitlines():
            utt_id, text = line.split(" ", 1)
            audio_path = trans_file.parent / f"{utt_id}.flac"
            yield audio_path, text


def load_model(backend: str, model_name: str):
    if backend == "whisper_trt":
        from whisper_trt import load_trt_model
        return load_trt_model(model_name)
    from whisper import load_model as load_pt_model
    return load_pt_model(model_name)


def main():
    args = parse_args()

    model = load_model(args.backend, args.model)
    normalizer = EnglishTextNormalizer()

    utterances = list(iter_utterances(Path(args.data_root)))
    if not utterances:
        raise SystemExit(f"No .trans.txt files found under {args.data_root}")
    if args.limit:
        utterances = utterances[: args.limit]

    refs, hyps = [], []
    t0 = time.perf_counter()
    for audio_path, ref_text in utterances:
        result = model.transcribe(str(audio_path))
        hyps.append(normalizer(result["text"]))
        refs.append(normalizer(ref_text))
    elapsed = time.perf_counter() - t0

    row = {
        "backend": args.backend,
        "model": args.model,
        "n_utterances": len(utterances),
        "wer": jiwer.wer(refs, hyps),
        "total_seconds": elapsed,
        "seconds_per_utterance": elapsed / len(utterances),
    }
    print(json.dumps(row, indent=2))

    if args.results_json:
        out = Path(args.results_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = json.loads(out.read_text()) if out.exists() else []
        rows.append(row)
        out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
