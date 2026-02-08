import os
import argparse
import subprocess
import sys
import shutil

def main():
    parser = argparse.ArgumentParser(description="CARD Audio2Script and Summarizer")
    parser.add_argument("--input", required=True, help="Path to input podcast audio")
    parser.add_argument("--device", default="cuda", help="Device to run diarization on (cuda/cpu)")
    parser.add_argument("--openai-key", help="OpenAI API Key")
    parser.add_argument("--voice-dir", default="stage2_voices", help="Directory for speaker samples")
    
    args = parser.parse_args()

    # 1. API KEY CHECK
    api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] No OpenAI API Key found. Use --openai-key")
        return

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        return

    current_env = os.environ.copy()
    try:
        # Find where python keeps packages
        site_packages = next(p for p in sys.path if 'site-packages' in p)
        nvidia_path = os.path.join(site_packages, 'nvidia')
        
        cudnn_lib = os.path.join(nvidia_path, 'cudnn', 'lib')
        cublas_lib = os.path.join(nvidia_path, 'cublas', 'lib')
        
        # If the folders exist, force the system to look there first
        if os.path.exists(cudnn_lib):
            # print(f"[INFO] Injecting NVIDIA lib paths to prevent crash...")
            current_ld = current_env.get('LD_LIBRARY_PATH', '')
            current_env['LD_LIBRARY_PATH'] = f"{cudnn_lib}:{cublas_lib}:{current_ld}"
    except Exception as e:
        print(f"[WARN] Could not auto-detect NVIDIA libs (might crash): {e}")

    # ==========================================
    # STAGE 1: Diarization
    # ==========================================
    print("\n" + "="*50)
    print(f"🚀 STARTING STAGE 1: Audio2Script")
    print("="*50)
    
    try:
        # We pass 'env=current_env' to ensure the child process sees the libraries
        subprocess.run([
            sys.executable, "diarize.py",
            "-a", input_path,
            "--device", args.device,
            "--batch-size", "2" 
        ], check=True, env=current_env)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Stage 1 crashed with code {e.returncode}")
        print("Tip: If code is -6 or -11, it's a library/driver mismatch.")
        return

    base_name = os.path.splitext(input_path)[0]
    diarization_json = f"{base_name}.json"

    if not os.path.exists(diarization_json):
        print(f"[ERROR] Expected output not found: {diarization_json}")
        print("Did you update diarize.py to export JSON?")
        return

    print(f"[SUCCESS] Stage 1 Complete. Output: {diarization_json}")

    # ==========================================
    # STAGE 2: Summarizer
    # ==========================================
    print("\n" + "="*50)
    print(f"🚀 STARTING STAGE 2: Summarizer")
    print("="*50)

    summary_output = f"{base_name}_summary.json"

    try:
        # We also pass 'env=current_env' here just in case torch is used
        subprocess.run([
            sys.executable, "summarizer.py",
            "--transcript", diarization_json,
            "--voice-dir", args.voice_dir,
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