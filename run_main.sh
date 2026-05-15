#!/bin/bash
# run_main.sh
# Launch the FAST denoising pipeline in a session-independent systemd scope
# so display/GDM crashes cannot kill the process.
#
# Usage:
#   bash run_main.sh                          # start pipeline
#   bash run_main.sh --config my_config.json  # use alternate config
#   bash run_main.sh --attach                 # attach to running tmux session
#   bash run_main.sh --status                 # show current pipeline status

set -euo pipefail

FAST_DIR="/home/schollab-gaga/Documents/FAST"
CONDA_ENV="FAST"
TMUX_SESSION="fast"
LOG_DIR="$FAST_DIR/logs"

# ── parse args ────────────────────────────────────────────────────────────────

CONFIG_ARG=""
MODE="start"

for arg in "$@"; do
	case "$arg" in
		--attach)  MODE="attach" ;;
		--status)  MODE="status" ;;
		--config*) CONFIG_ARG="$arg" ;;
	esac
done

# ── status mode ───────────────────────────────────────────────────────────────

if [ "$MODE" = "status" ]; then
	echo "── Pipeline status ──────────────────────────────────"
	cat "$LOG_DIR/_pipeline_status.json" 2>/dev/null || echo "No status file found"
	echo ""
	echo "── Last 20 log lines ────────────────────────────────"
	tail -20 "$(ls -t "$LOG_DIR"/_pipeline_log_*.txt 2>/dev/null | head -1)" 2>/dev/null \
		|| echo "No log file found"
	echo ""
	echo "── Running processes ────────────────────────────────"
	pgrep -af "python.*main.py" || echo "Pipeline is not running"
	exit 0
fi

# ── attach mode ───────────────────────────────────────────────────────────────

if [ "$MODE" = "attach" ]; then
	tmux attach -t "$TMUX_SESSION" || echo "No tmux session named '$TMUX_SESSION' found"
	exit 0
fi

# ── start mode ────────────────────────────────────────────────────────────────

# Guard: don't start a second instance
if pgrep -af "python.*main.py" > /dev/null 2>&1; then
	echo "Pipeline is already running:"
	pgrep -af "python.*main.py"
	echo ""
	echo "To attach to it:  bash run_main.sh --attach"
	echo "To check status:  bash run_main.sh --status"
	exit 1
fi

# Enable linger so user processes survive display/session crashes
# Safe to call repeatedly — idempotent
loginctl enable-linger "$USER"

mkdir -p "$LOG_DIR"

echo "Starting FAST pipeline..."
echo "  Config:  ${CONFIG_ARG:-$FAST_DIR/pipeline_config.json}"
echo "  Log dir: $LOG_DIR"
echo "  Session: tmux:$TMUX_SESSION (systemd user scope)"
echo ""

# Launch inside a detached tmux session, wrapped in a systemd user scope.
# The systemd scope places the process outside the login session cgroup
# so GDM/GNOME crashes cannot kill it.
systemd-run --user --scope --unit=fast-pipeline \
	tmux new-session -d -s "$TMUX_SESSION" \
	"bash -c 'source ~/.bashrc && conda activate $CONDA_ENV && \
	    python $FAST_DIR/main.py $CONFIG_ARG \
	    2>&1 | tee -a $LOG_DIR/nohup.out; \
	    echo \"Pipeline exited with code \$?\" >> $LOG_DIR/nohup.out'"

sleep 1

# Confirm it started
if pgrep -af "python.*main.py" > /dev/null 2>&1; then
	echo "Pipeline started successfully."
	echo ""
	echo "Useful commands:"
	echo "  Attach to live output:  bash run_main.sh --attach"
	echo "  Check status:           bash run_main.sh --status"
	echo "  Follow log file:        tail -f \$(ls -t $LOG_DIR/_pipeline_log_*.txt | head -1)"
	echo "  Detach from tmux:       Ctrl+B then D"
else
	echo "ERROR: Pipeline failed to start. Check:"
	echo "  cat $LOG_DIR/nohup.out"
	exit 1
fi