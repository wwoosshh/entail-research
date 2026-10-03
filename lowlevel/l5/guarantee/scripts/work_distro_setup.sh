set -e
id <user> 2>/dev/null || useradd -m -u 1000 -s /bin/bash -G sudo <user>
echo "<user> ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/<user>
printf '[user]\ndefault=<user>\n' > /etc/wsl.conf
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential python3.12-venv python3.12-dev git > /dev/null
python3 --version; gcc --version | head -1; ls /usr/lib/wsl/lib | head -3
