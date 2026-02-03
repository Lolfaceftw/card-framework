cd /Users/sean/Main/code/UCL/audio-splitter
source /Users/sean/Main/code/venv/UCL/audio-splitter/bin/activate

# Fix NumPy
pip install "numpy<2.0"

# Reinstall compatible versions
pip install --upgrade torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install --upgrade "transformers>=4.35.0"
pip install --upgrade speechbrain
pip install openai-whisper

# Test
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"