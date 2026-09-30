# ============================================================================
# FILE: models/local_llm/llama318b.py
# Local helper script for preparing an Ollama Llama 3.1 8B endpoint.
#
# Purpose:
# - Makes it easier to run a local LLM backend for experiments that use Ollama.
#
# Workflow:
# - Checks whether Ollama is already running.
# - Starts the server if needed.
# - Pulls the target model if missing.
# - Warms the model up before interactive use.
#
# Use this file when:
# - You want to use a local Layer 3 or local LLM-backed baseline flow.
# ============================================================================
import subprocess
import sys
import time
import requests

MODEL_NAME = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

def check_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except:
        return False

def start_ollama():
    print("🚀 Starting Ollama server...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    for i in range(10):
        time.sleep(1)
        if check_ollama_running():
            print("✅ Ollama server is running at http://localhost:11434")
            return True
        print(f"   Waiting... ({i+1}/10)")
    print("❌ Failed to start Ollama server")
    return False

def pull_model_if_needed():
    print(f"🔍 Checking model: {MODEL_NAME}")
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
    models = [m["name"] for m in r.json().get("models", [])]

    if not any(MODEL_NAME in m for m in models):
        print(f"📥 Pulling {MODEL_NAME} (Llama-3.1-8B-Instruct)...")
        subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
        print(f"✅ Model {MODEL_NAME} ready")
    else:
        print(f"✅ Model {MODEL_NAME} already exists")

def warmup_model():
    """Warm up model để lần gọi đầu không bị chậm"""
    print(f"🔥 Warming up {MODEL_NAME}...")
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": "hi", "stream": False},
            timeout=60
        )
        if r.status_code == 200:
            print("✅ Model warmed up and loaded into VRAM")
    except Exception as e:
        print(f"⚠️  Warmup failed: {e}")

def main():
    if check_ollama_running():
        print("✅ Ollama already running")
    else:
        if not start_ollama():
            sys.exit(1)

    pull_model_if_needed()
    warmup_model()

    print(f"\n{'='*50}")
    print(f"  Local LLM Server Ready")
    print(f"  Model   : {MODEL_NAME} (Llama-3.1-8B-Instruct)")
    print(f"  Endpoint: {OLLAMA_BASE_URL}/v1")
    print(f"{'='*50}")
    print("  Press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")

if __name__ == "__main__":
    main()
