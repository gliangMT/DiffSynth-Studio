#!/usr/bin/env bash
set -euo pipefail

########################################
# 用户按实际环境修改这里
########################################
WORK_DIR=$(pwd)
HOSTFILE="examples/wanvideo/model_training/full/wan2.2_multi/hostfile"
MASTER_PORT="29501"
NODE_SCRIPT="examples/wanvideo/model_training/full/wan2.2_multi/Wan2.2-I2V-A14B-torchrun-multi.sh"  # relative path

########################################
# 读取 hostfile
########################################
if [[ ! -f "${HOSTFILE}" ]]; then
    echo "[ERROR] hostfile not found: ${HOSTFILE}"
    exit 1
fi

mapfile -t HOSTS < <(
    awk '
        NF > 0 && $1 !~ /^#/ {
            print $1
        }
    ' "${HOSTFILE}" | awk '!seen[$0]++'
)

if [[ ${#HOSTS[@]} -eq 0 ]]; then
    echo "[ERROR] no valid hosts found in ${HOSTFILE}"
    exit 1
fi

NUM_MACHINES=${#HOSTS[@]}
MASTER_ADDR="${HOSTS[0]}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

########################################
# 打印信息
########################################
echo "[INFO] WORK_DIR=${WORK_DIR}"
echo "[INFO] TIMESTAMP=${TIMESTAMP}"
echo "[INFO] HOSTFILE=${HOSTFILE}"
echo "[INFO] HOSTS=${HOSTS[*]}"
echo "[INFO] NUM_MACHINES=${NUM_MACHINES}"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR}"
echo "[INFO] MASTER_PORT=${MASTER_PORT}"
echo "[INFO] NODE_SCRIPT=${NODE_SCRIPT}"

########################################
# 启动函数
########################################
launch_one_host() {
    local host="$1"
    local rank="$2"

    echo "[INFO] Launching host=${host}, machine_rank=${rank} ..."

    ssh "${host}" "bash -lc '
        set -euo pipefail
        cd \"${WORK_DIR}\"
        mkdir -p logs/launcher

        nohup bash \"${NODE_SCRIPT}\" \
            \"${rank}\" \
            \"${MASTER_ADDR}\" \
            \"${MASTER_PORT}\" \
            \"${TIMESTAMP}\" \
            > /dev/null 2>&1 &

        echo \$! > \"logs/launcher/launch_host${rank}_${TIMESTAMP}.pid\"
        echo \"[REMOTE] host=${host} machine_rank=${rank} launched, pid=\$(cat logs/launcher/launch_host${rank}_${TIMESTAMP}.pid)\"
    '"
}

########################################
# 先启动非 master，再启动 master
########################################
if (( NUM_MACHINES > 1 )); then
    echo "[INFO] Launching worker hosts first..."
    for (( rank=1; rank<NUM_MACHINES; rank++ )); do
        launch_one_host "${HOSTS[$rank]}" "${rank}"
    done
fi

echo "[INFO] Launching master host last..."
launch_one_host "${HOSTS[0]}" "0"

echo "[INFO] All launch commands have been issued."