import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="CARD Pipeline Wrapper")
    parser.add_argument("--input", required=True, help="Path to the full podcast audio file")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda/cpu)")
    parser.add_argument("--batch-size", default="2", help="Reduce to 1 if crashing, increase if hardware is good")
    args = parser.parse_args()

    # 1. Path Handling
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    print("--- STARTING CARD AUDIO PIPELINE ---")
    print(f"Input: {input_path}")
    print(f"Hardware Mode: {args.device} (Batch: {args.batch_size})")

    # 2. DYNAMIC LIBRARY FIX (The "Crash Preventer")
    # We prepare the environment variables BEFORE running the script
    current_env = os.environ.copy()
    try:
        # Find site-packages
        site_packages = next(p for p in sys.path if 'site-packages' in p)
        nvidia_path = os.path.join(site_packages, 'nvidia')
        
        cudnn_lib = os.path.join(nvidia_path, 'cudnn', 'lib')
        cublas_lib = os.path.join(nvidia_path, 'cublas', 'lib')
        
        if os.path.exists(cudnn_lib):
            print(f"[INFO] Injecting cuDNN lib path: {cudnn_lib}")
            # Add to LD_LIBRARY_PATH in the environment dict
            current_ld = current_env.get('LD_LIBRARY_PATH', '')
            current_env['LD_LIBRARY_PATH'] = f"{cudnn_lib}:{cublas_lib}:{current_ld}"
    except Exception as e:
        print(f"[WARN] Could not auto-detect NVIDIA libs: {e}")

    # 3. Construct the command
    cmd = [
        sys.executable, "diarize.py",
        "-a", input_path,
        "--device", args.device,
        "--batch-size", args.batch_size,
        "--suppress_numerals"
    ]

    # 4. Execution
    try:
        # Pass 'env=current_env' so the child process sees the libraries immediately
        subprocess.run(cmd, check=True, env=current_env)
        print("\n[SUCCESS] Processing complete.")
        print("Check the input folder for .txt and .srt outputs.")
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILURE] The pipeline crashed with error code {e.returncode}.")
        print("Tip: If it was an OOM (Out of Memory) error, try running with --batch-size 1")

if __name__ == "__main__":
    main()