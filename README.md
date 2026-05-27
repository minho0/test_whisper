# Faster-Whisper WAV Test

## Setup

```bash

python3 -m venv .venv
source .venv/bin/activate

pip install faster-whisper
```

## Download Model

Thor에서 사용하던 모델과 동일한 revision을 다운로드합니다.

```bash
mkdir -p models

python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dropbox-dash/faster-whisper-large-v3-turbo",
    revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    local_dir="./models/large-v3-turbo",
)
PY
```

## Run

```bash
python3 test_whisper.py your_wav_path --model_path your_model_path 
```

출력 예시:

```text
Bring me a coke from the cabinet.
```
