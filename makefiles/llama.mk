# -------------------------
# Paths
# -------------------------
# PROJECT_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
MODEL_DIR ?= $(PROJECT_ROOT)/models


# -------------------------
# llama.cpp Docker runtime
# -------------------------
LLAMA_IMAGE ?= ghcr.io/ggml-org/llama.cpp:server-cuda13
LLAMA_PORT ?= 8080

# Safer first model
E4B_MODEL ?= $(MODEL_DIR)/gemma-4-e4b-it.Q4_K_M.gguf
E4B_CTX ?= 4096
E4B_BATCH ?= 512
E4B_NGL ?= 99

# Larger model
LARGE_MODEL ?= $(MODEL_DIR)/gemma-4-26b-a4b-it.Q4_K_M.gguf
LARGE_CTX ?= 2048
LARGE_BATCH ?= 256
LARGE_NGL ?= 99

LLAMA_CONTAINER ?= goliath-llama

.PHONY: llama-pull llama-run-e4b llama-run-large llama-stop llama-logs llama-health

llama-pull:
	@echo "[make] Pulling llama.cpp Docker image..."
	docker pull $(LLAMA_IMAGE)

llama-run-e4b:
	@echo "[make] Starting llama.cpp server with E4B model..."
	docker rm -f $(LLAMA_CONTAINER) 2>/dev/null || true
	docker run -d \
		--name $(LLAMA_CONTAINER) \
		--gpus all \
		-p $(LLAMA_PORT):$(LLAMA_PORT) \
		-v $(MODEL_DIR):/models \
		$(LLAMA_IMAGE) \
		-m /models/$(notdir $(E4B_MODEL)) \
		--host 0.0.0.0 \
		--port $(LLAMA_PORT) \
		-ngl $(E4B_NGL) \
		-c $(E4B_CTX) \
		-b $(E4B_BATCH)

llama-run-large:
	@echo "[make] Starting llama.cpp server with large model..."
	docker rm -f $(LLAMA_CONTAINER) 2>/dev/null || true
	docker run -d \
		--name $(LLAMA_CONTAINER) \
		--gpus all \
		-p $(LLAMA_PORT):$(LLAMA_PORT) \
		-v $(MODEL_DIR):/models \
		$(LLAMA_IMAGE) \
		-m /models/$(notdir $(LARGE_MODEL)) \
		--host 0.0.0.0 \
		--port $(LLAMA_PORT) \
		-ngl $(LARGE_NGL) \
		-c $(LARGE_CTX) \
		-b $(LARGE_BATCH)

llama-stop:
	@echo "[make] Stopping llama.cpp container..."
	docker rm -f $(LLAMA_CONTAINER) 2>/dev/null || true

llama-logs:
	@echo "[make] Showing llama.cpp logs..."
	docker logs -f $(LLAMA_CONTAINER)

llama-health:
	@echo "[make] Checking llama.cpp health..."
	curl http://127.0.0.1:$(LLAMA_PORT)/health

print-paths:
	@echo "PROJECT_ROOT=$(PROJECT_ROOT)"
	@echo "MODEL_DIR=$(MODEL_DIR)"
	@echo "E4B_MODEL=$(E4B_MODEL)"

docker-stop-all:
	@echo "[make] Stopping ALL running docker containers..."
	@if [ -n "$$(docker ps -q)" ]; then \
		docker stop $$(docker ps -q); \
	else \
		echo "[make] No running containers."; \
	fi

docker-rm-all:
	@echo "[make] Removing ALL containers..."
	@if [ -n "$$(docker ps -aq)" ]; then \
		docker rm -f $$(docker ps -aq); \
	else \
		echo "[make] No containers to remove."; \
	fi

docker-reset:
	@echo "[make] FULL Docker reset..."
	@if [ -n "$$(docker ps -q)" ]; then \
		docker stop $$(docker ps -q); \
	fi
	@if [ -n "$$(docker ps -aq)" ]; then \
		docker rm -f $$(docker ps -aq); \
	fi