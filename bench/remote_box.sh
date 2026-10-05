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
SSH=(ssh -p "$port" -i "$key" -o StrictHostKeyChecking=accept-new "root@$host")
remote_dir="/root/candle-fused-attn"

case "$cmd" in
  send)
    bundle="target/remote/candle-fused-attn.bundle"
    mkdir -p target/remote
    git bundle create "$bundle" --all
    echo "bundle: $(du -h "$bundle" | cut -f1), HEAD $(git rev-parse --short HEAD)"
    scp -P "$port" -i "$key" "$bundle" "root@$host:/root/candle-fused-attn.bundle"
    "${SSH[@]}" "set -e
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
    "${SSH[@]}" "cd $remote_dir && mkdir -p target/box/$label &&
      setsid nohup env $envs bash bench/box.sh $label > target/box/$label.out 2>&1 < /dev/null &
      sleep 2; echo started; tail -3 target/box/$label.out"
    ;;
  status)
    "${SSH[@]}" "cd $remote_dir && tail -15 target/box/$label.out;
      if grep -q 'done: target/box' target/box/$label.out; then echo FINISHED;
      elif pgrep -f 'bench/box.sh $label' > /dev/null; then echo RUNNING; else echo STOPPED-EARLY; fi"
    ;;
  pull)
    mkdir -p target/box-home
    scp -r -P "$port" -i "$key" "root@$host:$remote_dir/target/box/$label" target/box-home/
    scp -P "$port" -i "$key" "root@$host:$remote_dir/target/box/$label.out" "target/box-home/$label/"
    echo "home: target/box-home/$label ($(ls target/box-home/$label | wc -l) entries)"
    ;;
  *) echo "unknown command $cmd" >&2; exit 1 ;;
esac
