#!/usr/bin/env bash
# Runs bench/box.sh on a rented box from this machine (devlog FA17), without publishing anything:
# the local repository travels as a git bundle (every branch and tag), so commits not yet pushed
# are measured exactly as they are here.
#
# Usage, from the crate root:
#   bash bench/remote_box.sh <host> <port> <label> send     # bundle + scp + rustup + clone (CPU, ~3 min)
#   bash bench/remote_box.sh <host> <port> <label> run      # starts box.sh detached (GPU ~5 min after ~3 min of builds)
#   bash bench/remote_box.sh <host> <port> <label> status   # the last lines of its log, and whether it has finished
#   bash bench/remote_box.sh <host> <port> <label> pull     # target/box/<label>/ -> target/box-home/<label>/
# Extra environment for box.sh (BASE, ROUNDS) passes through `run`.
# One attempt per command: if the box refuses SSH, look at the dashboard, do not loop.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
host="${1:?host}"; port="${2:?port}"; label="${3:?label}"; cmd="${4:?send|run|status|pull}"
key="$HOME/.ssh/id_ed25519"
OPTS=(-i "$key" -o StrictHostKeyChecking=accept-new)
SSH=(ssh -p "$port" "${OPTS[@]}" "root@$host")
SCP=(scp -P "$port" "${OPTS[@]}")
remote_dir="/root/candle-fused-attn"

case "$cmd" in
  send)
    bundle="target/remote/candle-fused-attn.bundle"
    mkdir -p target/remote
    git bundle create "$bundle" --all
    echo "bundle: $(du -h "$bundle" | cut -f1), HEAD $(git rev-parse --short HEAD)"
    "${SCP[@]}" "$bundle" "root@$host:/root/candle-fused-attn.bundle"
    "${SSH[@]}" "set -e
      # vast's PyTorch image ships torch in /venv/main but not always safetensors (FA12, FA17).
      /venv/main/bin/python -c 'import safetensors' 2> /dev/null \
        || /venv/main/bin/python -m pip install -q safetensors
      if [ ! -x \$HOME/.cargo/bin/cargo ]; then
        curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
      fi
      rm -rf $remote_dir
      git clone -q /root/candle-fused-attn.bundle $remote_dir
      cd $remote_dir && git checkout -q $(git rev-parse --abbrev-ref HEAD)
      echo \"box at \$(git describe --always --tags), tags: \$(git tag | tr '\n' ' ')\""
    ;;
  run)
    envs="BASE=${BASE:-v0.3.0} ROUNDS=${ROUNDS:-6}"
    # cd first, on its own line: a trailing `&` backgrounds a whole `a && b && c` list, which left
    # the pid and the tail in the wrong directory on the first 5090 run.
    "${SSH[@]}" "cd $remote_dir || exit 1
      mkdir -p target/box/$label
      setsid nohup env $envs bash bench/box.sh $label > target/box/$label.out 2>&1 < /dev/null &
      echo \$! > target/box/$label.pid; sleep 5; echo started; tail -3 target/box/$label.out"
    ;;
  status)
    "${SSH[@]}" "cd $remote_dir && tail -15 target/box/$label.out;
      if grep -q 'done: target/box' target/box/$label.out; then echo FINISHED;
      elif kill -0 \$(cat target/box/$label.pid) 2> /dev/null; then echo RUNNING; else echo STOPPED-EARLY; fi"
    ;;
  pull)
    # Everything but the bulk: the two bitwise dumps (~0.5 GB each; bitwise.txt holds the verdict)
    # and the shared inputs / output tensors (~0.5 GB) stay on the box. The 5090's full copy was
    # 1.8 GB and took ~10 min home.
    mkdir -p "target/box-home/$label"
    "${SSH[@]}" "cd $remote_dir/target/box && tar czf - --exclude='$label/bitwise-base' \
      --exclude='$label/bitwise-head' --exclude='*.safetensors' $label $label.out" \
      | tar xzf - -C target/box-home
    echo "home: target/box-home/$label ($(ls target/box-home/$label | wc -l) entries)"
    ;;
  *) echo "unknown command $cmd" >&2; exit 1 ;;
esac
