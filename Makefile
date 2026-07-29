# ─────────────────────────────────────────────────────────────────────────────
#  Makefile — Jetson WhisperTRT developer shortcuts
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help build build-dev shell run validate list-devices transcribe-file benchmark validate-librispeech export-onnx live clean clean-docker

DOCKER_IMAGE  := jetson-whisper-trt:latest
DOCKER_DEV    := jetson-whisper-trt:dev
COMPOSE       := docker compose
MODEL         ?= base.en
BACKEND       ?= whisper_trt
CACHE_DIR     := /opt/models/jetson-whisper-trt
DATASETS      ?= /opt/datasets

help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Docker ────────────────────────────────────────────────────────────────────
build:          ## Build the production Docker image
	mkdir -p $(CACHE_DIR)
	$(COMPOSE) build pipeline

build-dev:      ## Build the development Docker image
	mkdir -p $(CACHE_DIR)
	$(COMPOSE) build dev

shell:          ## Open an interactive shell inside the dev container
	$(COMPOSE) run --rm dev bash

# ── Pipeline ──────────────────────────────────────────────────────────────────
run:            ## Run the live transcription pipeline (production image)
	$(COMPOSE) run --rm pipeline

live:           ## Run the live transcription pipeline inside the dev container
	$(COMPOSE) run --rm dev python scripts/live_transcribe.py --config configs/pipeline.yaml

validate:       ## Verify environment with scripts/validate_install.py
	$(COMPOSE) run --rm dev python scripts/validate_install.py

list-devices:   ## List PortAudio input devices visible inside the container
	$(COMPOSE) run --rm dev python -c "\
import pyaudio; p = pyaudio.PyAudio(); info = p.get_host_api_info_by_index(0); \
[print(i, p.get_device_info_by_host_api_device_index(0, i).get('name'), \
 'in_ch=' + str(p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels'))) \
 for i in range(info.get('deviceCount'))]"

# ── Transcription / benchmarking ──────────────────────────────────────────────
transcribe-file: ## Transcribe a WAV file (AUDIO=path, MODEL=base.en, BACKEND=whisper_trt)
	$(COMPOSE) run --rm dev python scripts/transcribe_file.py $(AUDIO) --model $(MODEL) --backend $(BACKEND)

benchmark:      ## Compare whisper / whisper_trt latency+memory (AUDIO=path, MODEL=base.en)
	$(COMPOSE) run --rm dev python scripts/benchmark.py --audio $(AUDIO) --model $(MODEL)

validate-librispeech: ## WER on LibriSpeech test-clean (MODEL=base.en, LIMIT=100, BACKEND=whisper_trt)
	$(COMPOSE) run --rm \
	  -v $(DATASETS)/librispeech:/dataset/librispeech:ro \
	  dev python scripts/validate_librispeech.py \
	  --model $(MODEL) \
	  --backend $(BACKEND) \
	  $(if $(LIMIT),--limit $(LIMIT),) \
	  --results-json output/librispeech_results.json

export-onnx:    ## Export encoder+decoder ONNX for MODEL=base.en to /opt/models/jetson-whisper-trt/onnx
	$(COMPOSE) run --rm \
	  -v $(CACHE_DIR)/onnx:/opt/models/jetson-whisper-trt/onnx \
	  dev python scripts/export_onnx.py --model $(MODEL)

# ── Housekeeping ──────────────────────────────────────────────────────────────
clean:          ## Remove __pycache__, .pyc (does NOT touch /opt/models — shared storage)
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null; true
	find . -name '*.pyc' -delete 2>/dev/null; true

clean-docker:   ## Remove all project Docker images
	docker rmi $(DOCKER_IMAGE) $(DOCKER_DEV) 2>/dev/null; true
