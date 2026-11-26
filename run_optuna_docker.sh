#!/usr/bin/env bash
set -euo pipefail

study_name="${STUDY_NAME:-molearn_simrna_distance_matrix_converged}"
gpu_ids="${GPU_IDS:-}"
n_trials="${N_TRIALS:-10}"
max_epochs="${MAX_EPOCHS:-${EPOCHS:-500}}"
patience="${PATIENCE:-40}"
frame_step="${FRAME_STEP:-1}"
development_frames="${DEVELOPMENT_FRAMES:-7007}"
validation_split="${VALIDATION_SPLIT:-0.1}"
simrna_sp_weight="${SIMRNA_SP_WEIGHT:-0.0}"
postgres_password="${POSTGRES_PASSWORD:-molearn_optuna_local}"
network_name="molearn-optuna"
database_container="molearn-optuna-postgres"
base_image="molearn-rna:latest"
worker_image="molearn-rna-optuna:latest"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_directory="${DATA_DIRECTORY:-${repository_root}/yjdf_simrna}"
results_directory="${RESULTS_DIRECTORY:-${repository_root}/optuna_results}"

if [[ ! -d "${data_directory}/apo" || ! -d "${data_directory}/holo" ]]; then
    echo "Missing apo/holo data under: ${data_directory}" >&2
    exit 1
fi
mkdir -p "${results_directory}"

if [[ -z "${gpu_ids}" ]]; then
    gpu_ids="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
if [[ -z "${gpu_ids}" ]]; then
    echo "No GPUs detected. Set GPU_IDS explicitly, for example GPU_IDS=0." >&2
    exit 1
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    docker build -t "${base_image}" "${repository_root}"
    docker build -f "${repository_root}/Dockerfile.optuna" \
        -t "${worker_image}" "${repository_root}"
fi

docker network inspect "${network_name}" >/dev/null 2>&1 || \
    docker network create "${network_name}" >/dev/null

if ! docker container inspect "${database_container}" >/dev/null 2>&1; then
    docker run -d --name "${database_container}" \
        --network "${network_name}" \
        -e POSTGRES_USER=optuna \
        -e "POSTGRES_PASSWORD=${postgres_password}" \
        -e POSTGRES_DB=optuna \
        -v molearn-optuna-postgres-data:/var/lib/postgresql/data \
        postgres:16 >/dev/null
else
    docker start "${database_container}" >/dev/null
fi

echo "Waiting for PostgreSQL..."
until docker exec "${database_container}" pg_isready -U optuna -d optuna >/dev/null 2>&1; do
    sleep 2
done

storage_url="postgresql+psycopg2://optuna:${postgres_password}@${database_container}:5432/optuna"
container_study="${study_name//[^[:alnum:]_.-]/-}"
IFS=',' read -ra gpu_array <<< "${gpu_ids}"
for gpu_id in "${gpu_array[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    gpu_arguments=()
    if [[ "${gpu_id}" == "cpu" ]]; then
        gpu_request="CPU"
    elif [[ "${gpu_id}" == "all" ]]; then
        gpu_request="all"
        gpu_arguments=(--gpus "${gpu_request}")
    else
        gpu_request="device=${gpu_id}"
        gpu_arguments=(--gpus "${gpu_request}")
    fi
    worker_name="molearn-optuna-${container_study}-gpu${gpu_id}"
    if docker container inspect "${worker_name}" >/dev/null 2>&1; then
        echo "Container ${worker_name} already exists; remove it or choose another STUDY_NAME." >&2
        exit 1
    fi
    docker run -d --name "${worker_name}" \
        --network "${network_name}" \
        "${gpu_arguments[@]}" \
        --user "$(id -u):$(id -g)" \
        -e MPLCONFIGDIR=/tmp/matplotlib \
        -v "${data_directory}:/data/yjdf_simrna:ro" \
        -v "${results_directory}:/output" \
        "${worker_image}" \
        python /opt/molearn/examples/optuna_simrna_RNA.py \
          --storage "${storage_url}" \
          --study-name "${study_name}" \
          --n-trials "${n_trials}" \
          --max-epochs "${max_epochs}" \
          --patience "${patience}" \
          --frame-step "${frame_step}" \
          --development-frames "${development_frames}" \
          --validation-split "${validation_split}" \
          --simrna-sp-weight "${simrna_sp_weight}" \
          --data-root /data \
          --output-root /output \
          --ensemble-config /opt/molearn/examples/simrna_ensembles.json
    echo "Started ${worker_name} on ${gpu_request}"
done

echo "Study: ${study_name}"
echo "Stopping: validation convergence (patience=${patience}), max epochs=${max_epochs}"
echo "Results: ${results_directory}"
echo "Follow a worker with: docker logs -f molearn-optuna-${container_study}-gpu${gpu_array[0]}"
