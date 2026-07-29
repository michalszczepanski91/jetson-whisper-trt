#!/usr/bin/env python3
"""Export Whisper's audio encoder and text decoder blocks to ONNX.

whisper_trt only ever builds ONNX in-memory as an intermediate step toward a
TensorRT engine (inside torch2trt.torch2trt(), via an io.BytesIO buffer) — it
never saves the file to disk. This reuses whisper_trt's own input-wrapper
modules (_AudioEncoderEngine, _TextDecoderEngine) and the exact dummy-input
shapes its WhisperTRTBuilder.build_audio_encoder_engine()/
build_text_decoder_engine() use internally, but calls torch.onnx.export()
directly and writes the result — for a portable artifact (plain onnxruntime,
no TensorRT/torch2trt needed) and to have a saved model under /opt/models,
matching jetson-yolov8-trt's scripts/export_onnx.py convention.

Two separate ONNX files, not one: Whisper's encoder and decoder are distinct
graphs (the decoder runs once per generated token, taking the encoder's
output as a fixed input) — there's no single combined forward pass to export.

Usage (inside dev container):
    python scripts/export_onnx.py --model base.en
"""

import argparse
from pathlib import Path

import torch
import whisper_trt.model as wtm
from whisper_trt.model import MODEL_BUILDERS, _AudioEncoderEngine, _TextDecoderEngine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["tiny.en", "base.en", "small.en"], default="base.en")
    p.add_argument("--output-dir", default="/opt/models/jetson-whisper-trt/onnx")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


@torch.no_grad()
def export_encoder(model, dims, output_path, opset):
    encoder_module = _AudioEncoderEngine(
        model.encoder.conv1, model.encoder.conv2, model.encoder.blocks, model.encoder.ln_post
    ).cuda().eval()

    n_frames = dims.n_audio_ctx * 2
    x = torch.randn(1, dims.n_mels, n_frames).cuda()
    positional_embedding = model.encoder.positional_embedding.cuda().detach()

    torch.onnx.export(
        encoder_module,
        (x, positional_embedding),
        str(output_path),
        input_names=["x", "positional_embedding"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,  # see jetson-whisper-trt's README — this PyTorch's default
                       # dynamo exporter is incompatible with how torch2trt/this
                       # export uses dynamic_axes-shaped tracing
    )


@torch.no_grad()
def export_decoder(model, dims, output_path, opset):
    decoder_module = _TextDecoderEngine(model.decoder.blocks).cuda().eval()

    x = torch.randn(1, 1, dims.n_text_state).cuda()
    xa = torch.randn(1, dims.n_audio_ctx, dims.n_audio_state).cuda()
    mask = torch.randn(dims.n_text_ctx, dims.n_text_ctx).cuda()

    torch.onnx.export(
        decoder_module,
        (x, xa, mask),
        str(output_path),
        input_names=["x", "xa", "mask"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = MODEL_BUILDERS[args.model]
    model = wtm.load_model(builder.model).cuda().eval()
    dims = model.dims

    stem = args.model.replace(".", "_")
    encoder_path = output_dir / f"{stem}_encoder.onnx"
    decoder_path = output_dir / f"{stem}_decoder.onnx"

    print(f"Exporting encoder -> {encoder_path}")
    export_encoder(model, dims, encoder_path, args.opset)
    print(f"Exporting decoder -> {decoder_path}")
    export_decoder(model, dims, decoder_path, args.opset)

    print(f"Saved: {encoder_path} ({encoder_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved: {decoder_path} ({decoder_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
