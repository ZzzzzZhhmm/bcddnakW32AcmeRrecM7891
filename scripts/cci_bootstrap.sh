#!/usr/bin/env bash
# CCI 容器重启后的一键自愈脚本。
#
# 背景：这台 CCI 容器（k8s pod）每隔数小时会整体重启，/root 与 /tmp 全部
# 重置，导致 ~/.ssh（GitHub SSH-over-443 配置）和 /root/.gitconfig
# （safe.directory、提交身份）丢失，git fetch/push 报 "port 22 timed out"
# 或 "dubious ownership"。
#
# 永久化策略（本脚本 + 仓库配置双保险）：
#   1. SSH：密钥与 ssh_config 常驻 AFS（${BACKUP_DIR}），仓库级
#      core.sshCommand 已固化在 .git/config（AFS 上，重启不丢），因此
#      在仓库目录内执行 git fetch/push 无需任何恢复即可工作；
#   2. safe.directory 与提交身份：git 出于安全设计只认 /root/.gitconfig
#      或环境变量，无法固化到仓库配置里，重启后需要重建——运行本脚本即可。
#
# 用法：容器重启后（或遇到任何 git 权限/连接报错时）执行一次：
#   bash /mnt/afs/task3_2/L202500276_lwz/projects/WARM/scripts/cci_bootstrap.sh

set -euo pipefail

PROJECT_DIR="/mnt/afs/task3_2/L202500276_lwz/projects/WARM"
BACKUP_DIR="/mnt/afs/task3_2/L202500276_lwz/projects/.warm_ssh_backup"
GIT_USER_NAME="howmean Z"
GIT_USER_EMAIL="203061685+ZzzzzZhhmm@users.noreply.github.com"

# 1. /root/.gitconfig：safe.directory（root 容器 + 用户属主仓库必需）与身份
git config --global --add safe.directory "${PROJECT_DIR}" 2>/dev/null || true
if ! git config --global --get-all safe.directory | grep -qx "${PROJECT_DIR}"; then
  echo "error: failed to register safe.directory for ${PROJECT_DIR}" >&2
  exit 1
fi
git config --global user.name "${GIT_USER_NAME}"
git config --global user.email "${GIT_USER_EMAIL}"

# 2. ~/.ssh：从 AFS 备份恢复（非必需——仓库级 core.sshCommand 已可独立工作，
#    恢复它只是让仓库目录之外的 ssh github.com 等命令也可用）
if [[ -f "${BACKUP_DIR}/id_ed25519" ]]; then
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  cp "${BACKUP_DIR}/id_ed25519" "${BACKUP_DIR}/id_ed25519.pub" ~/.ssh/
  cp "${BACKUP_DIR}/config" ~/.ssh/config 2>/dev/null || true
  chmod 600 ~/.ssh/id_ed25519 ~/.ssh/config 2>/dev/null || true
fi

# 3. 仓库级配置兜底（已在 .git/config 持久化，这里幂等重写以防被误删）
git -C "${PROJECT_DIR}" config core.sshCommand "ssh -F ${BACKUP_DIR}/ssh_config"
git -C "${PROJECT_DIR}" config user.name "${GIT_USER_NAME}"
git -C "${PROJECT_DIR}" config user.email "${GIT_USER_EMAIL}"

# 4. 自检
echo "== git identity =="
git -C "${PROJECT_DIR}" config user.name
git -C "${PROJECT_DIR}" config user.email
echo "== connectivity =="
if git -C "${PROJECT_DIR}" ls-remote --heads origin >/dev/null 2>&1; then
  echo "OK: git can reach origin ($(git -C "${PROJECT_DIR}" remote get-url origin))"
else
  echo "error: git still cannot reach origin; check ${BACKUP_DIR} contents" >&2
  exit 1
fi
echo "bootstrap complete."
