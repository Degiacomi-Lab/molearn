#!/usr/bin/env bash
set -euo pipefail

study_name="${STUDY_NAME:-molearn_simrna_distance_matrix}"
container_study="${study_name//[^[:alnum:]_.-]/-}"
for container in $(docker ps -aq --filter "name=molearn-optuna-${container_study}-gpu"); do
    docker stop "${container}" >/dev/null || true
done
echo "Stopped Optuna workers for study ${study_name}. PostgreSQL remains running."
