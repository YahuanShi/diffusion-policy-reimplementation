#!/bin/bash
# Auto-continue script for diffusion policy reimplementation
# Run in tmux: tmux new -s diffusion -d 'bash auto_continue.sh'

PROJECT_DIR="/mnt/4TBSSD/users/yahuanshi/Projects/il/diffusion-policy-reimplementation"
LOG_FILE="$PROJECT_DIR/auto_continue.log"
RETRY_INTERVAL=300  # 5 minutes between retries

cd "$PROJECT_DIR"

while true; do
    echo "[$(date)] Starting Claude session..." | tee -a "$LOG_FILE"

    claude -p "阅读 PROGRESS.md 和 CLAUDE.md，继续实现下一个未完成的步骤。完成后自动 commit 和 push（按 PROGRESS.md 中的日期 backdate），然后更新 PROGRESS.md 标记完成，继续下一步直到 session 用尽。" \
        --allowedTools "Edit,Read,Bash,Write" \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=$?
    echo "[$(date)] Session ended (exit=$EXIT_CODE). Waiting ${RETRY_INTERVAL}s..." | tee -a "$LOG_FILE"
    sleep $RETRY_INTERVAL
done
