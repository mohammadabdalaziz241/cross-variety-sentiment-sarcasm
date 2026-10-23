#!/usr/bin/env bash
# Publish the improved adapter model cards and the demo Space to Hugging Face.
# Run from the repository root after `hf auth login` (a token with write access).
set -euo pipefail

USER_NS="${HF_USER:-Mohammadeeu20}"
HERE="$(cd "$(dirname "$0")" && pwd)"

for v in en-uk en-au en-in; do
  echo "Updating model card for ${USER_NS}/besstie-sarcasm-gemma2b-lora-${v}"
  hf upload "${USER_NS}/besstie-sarcasm-gemma2b-lora-${v}" \
    "${HERE}/model_cards/README_${v}.md" README.md \
    --commit-message "Improve model card: cross-variety results, limitations, demo link"
done

SPACE="${USER_NS}/besstie-sarcasm-detector"
echo "Creating (or reusing) Space ${SPACE} and uploading files"
SPACE="${SPACE}" SPACE_DIR="${HERE}/space" python - <<'PY'
import os
from huggingface_hub import create_repo, upload_folder
repo = os.environ["SPACE"]
create_repo(repo, repo_type="space", space_sdk="gradio", exist_ok=True)
print(upload_folder(repo_id=repo, repo_type="space", folder_path=os.environ["SPACE_DIR"],
                    commit_message="Add sarcasm detector demo"))
PY

echo
echo "Done. Last step: open https://huggingface.co/spaces/${SPACE}/settings"
echo "and add a secret named HF_TOKEN (a read token from an account that accepted the Gemma licence)."
