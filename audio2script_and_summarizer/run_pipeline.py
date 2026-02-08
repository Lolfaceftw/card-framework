import os
import argparse
import subprocess
import sys
import shutil
import torch  # <--- Added to check for GPU

def main():
    # Detect hardware automatically
    # If users have NVIDIA drivers but no GPU, this safely falls back to CPU
    detected_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="CARD Audio2Script and Summarizer")
    parser.add_argument("--input", required=True, help="Path to input podcast audio")
    
    # [CHANGED] Now defaults to whatever torch detects, instead of hardcoded "cuda"
    parser.add_argument("--device", default=detected_device, help=f"Device to run on (auto-detected: {detected_device})")
    
    parser.add_argument("--api-key", help="LLM Provider API Key (Gemini, OpenAI, etc.)")
    parser.add_argument("--voice-dir", help="Optional override for voice directory")
    
    args = parser.parse_args()

    print(f"[INFO] Running on device: {args.device.upper()}")

    # 1. API KEY CHECK
    api_key = args.api_key or os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("[ERROR] No API Key found. Please set LLM_API_KEY or use --api-key")
        return

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        return

    # Env setup for WSL/Linux GPU support (Standard NVIDIA fix)
    current_env = os.environ.copy()
    try:
        site_packages = next(p for p in sys.path if 'site-packages' in p)
        nvidia_path = os.path.join(site_packages, 'nvidia')
        cudnn_lib = os.path.join(nvidia_path, 'cudnn', 'lib')
        cublas_lib = os.path.join(nvidia_path, 'cublas', 'lib')
        if os.path.exists(cudnn_lib):
            current_ld = current_env.get('LD_LIBRARY_PATH', '')
            current_env['LD_LIBRARY_PATH'] = f"{cudnn_lib}:{cublas_lib}:{current_ld}"
    except Exception:
        pass

    # ==========================================
    # PATH SETUP
    # ==========================================
    base_name = os.path.splitext(input_path)[0]
    diarization_json = f"{base_name}.json"
    
    if args.voice_dir:
        voice_dir = args.voice_dir
    else:
        voice_dir = f"{base_name}_voices"

    summary_output = f"{base_name}_summary.json"


    # ==========================================
    # STAGE 1: Diarization
    # ==========================================
    print("\n" + "="*50)
    print(f"🚀 STAGE 1: Diarization")
    print("="*50)
    
    # Determine batch size based on device
    # CPU cannot handle batching well, so we default to 1
    batch_size = "2" if args.device == "cuda" else "1"
    
    try:
        subprocess.run([
            sys.executable, "diarize.py",
            "-a", input_path,
            "--device", args.device,
            "--batch-size", batch_size 
        ], check=True, env=current_env)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Stage 1 crashed with code {e.returncode}")
        return

    if not os.path.exists(diarization_json):
        print(f"[ERROR] Expected output not found: {diarization_json}")
        return

    print(f"[SUCCESS] JSON saved to: {diarization_json}")

    # ==========================================
    # STAGE 1.5: Audio Splitting
    # ==========================================
    print("\n" + "="*50)
    print(f"✂️ STAGE 1.5: Audio Splitting")
    print(f"Output Dir: {voice_dir}")
    print("="*50)

    try:
        subprocess.run([
            sys.executable, "audio_splitter.py",
            "--audio", input_path,
            "--json", diarization_json,
            "--output-dir", voice_dir
        ], check=True, env=current_env)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Splitter crashed with code {e.returncode}")
        return

    # ==========================================
    # STAGE 2: Summarizer
    # ==========================================
    print("\n" + "="*50)
    print(f"🚀 STAGE 2: Summarizer")
    print("="*50)

    try:
        subprocess.run([
            sys.executable, "summarizer.py",
            "--transcript", diarization_json,
            "--voice-dir", voice_dir,
            "--output", summary_output,
            "--api-key", api_key 
        ], check=True, env=current_env)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Stage 2 crashed with code {e.returncode}")
        return

    print("\n" + "="*50)
    print(f"✅ PIPELINE COMPLETE. Summary saved to:")
    print(f"{summary_output}")
    print("="*50)

if __name__ == "__main__":
    main()