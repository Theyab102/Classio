#!/bin/sh
set -eu
case "$(uname -sm)" in "Linux x86_64") ;; *) echo "Install Gitleaks 8.28.0 for this platform using the official checksums." >&2; exit 1;; esac
classio_tools_dir="${1:-/tmp/classio-security-tools}"
mkdir -p "$classio_tools_dir"
curl -fLsS --max-time 60 -o "$classio_tools_dir/gitleaks.tar.gz" https://github.com/gitleaks/gitleaks/releases/download/v8.28.0/gitleaks_8.28.0_linux_x64.tar.gz
printf '%s  %s\n' "a65b5253807a68ac0cafa4414031fd740aeb55f54fb7e55f386acb52e6a840eb" "$classio_tools_dir/gitleaks.tar.gz" | sha256sum --check --status
tar -xzf "$classio_tools_dir/gitleaks.tar.gz" -C "$classio_tools_dir" gitleaks
"$classio_tools_dir/gitleaks" version
