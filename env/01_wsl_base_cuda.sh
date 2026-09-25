#!/usr/bin/env bash
# WSL2 Ubuntu 24.04: base dev packages + user + CUDA toolkit + Nsight.
# Run as root inside the distro:  bash 01_wsl_base_cuda.sh [username] [cuda-toolkit-pkg]
# NOTE: never install a Linux NVIDIA driver inside WSL; the Windows driver is used via /usr/lib/wsl/lib.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

USERNAME="${1:-<user>}"
CUDA_PKG="${2:-cuda-toolkit-13-0}"

echo "== [1/6] apt update/upgrade"
apt-get update -y
apt-get upgrade -y

echo "== [2/6] base dev packages"
apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git git-lfs pkg-config \
  python3 python3-venv python3-dev python3-pip \
  wget curl ca-certificates gnupg lsb-release \
  clang lld llvm libssl-dev zlib1g-dev libncurses-dev libxml2-dev \
  htop tmux unzip zip jq sudo vim less

echo "== [3/6] create user ${USERNAME} (passwordless sudo for a local dev VM)"
if ! id -u "${USERNAME}" >/dev/null 2>&1; then
  useradd -m -s /bin/bash -G sudo "${USERNAME}"
fi
echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/90-${USERNAME}"
chmod 0440 "/etc/sudoers.d/90-${USERNAME}"

cat > /etc/wsl.conf <<WSLCONF
[user]
default=${USERNAME}

[boot]
systemd=true
WSLCONF

echo "== [4/6] NVIDIA CUDA apt repo (wsl-ubuntu) + ${CUDA_PKG}"
KEYRING=/tmp/cuda-keyring_1.1-1_all.deb
wget -q -O "${KEYRING}" https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i "${KEYRING}"
apt-get update -y
apt-get install -y "${CUDA_PKG}"

echo "== [5/6] CUDA environment via /etc/profile.d/cuda.sh"
cat > /etc/profile.d/cuda.sh <<'CUDASH'
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
CUDASH
chmod 0644 /etc/profile.d/cuda.sh

echo "== [6/6] versions"
nvidia-smi || echo "WARN: nvidia-smi not available in WSL (check Windows driver / WSL GPU passthrough)"
/usr/local/cuda/bin/nvcc --version
ls /usr/local/cuda/bin | grep -E '^(ncu|nsys|cuobjdump|nvdisasm|ptxas|compute-sanitizer)$' || true
ls -d /usr/local/cuda-* || true
echo "DONE 01 base+cuda"
