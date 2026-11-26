#!/bin/bash
#SBATCH --job-name=molearn-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

MOLEARN_IMAGE="${MOLEARN_IMAGE:?Set MOLEARN_IMAGE to the .sif image path}"
MOLEARN_PROJECT="${MOLEARN_PROJECT:-${SLURM_SUBMIT_DIR}}"

if [[ -n "${MODELLER_LICENSE_FILE:-}" ]]; then
    if [[ ! -r "${MODELLER_LICENSE_FILE}" ]]; then
        echo "Cannot read MODELLER_LICENSE_FILE=${MODELLER_LICENSE_FILE}" >&2
        exit 1
    fi
    export APPTAINERENV_KEY_MODELLER="$(tr -d '\r\n' < "${MODELLER_LICENSE_FILE}")"
    export SINGULARITYENV_KEY_MODELLER="${APPTAINERENV_KEY_MODELLER}"
fi

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_RUNTIME="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_RUNTIME="$(command -v singularity)"
else
    echo "Neither apptainer nor singularity is available" >&2
    exit 1
fi

cd "${MOLEARN_PROJECT}"
srun "${CONTAINER_RUNTIME}" exec --nv \
    --bind "${MOLEARN_PROJECT}:/workspace/molearn" \
    "${MOLEARN_IMAGE}" \
    bash -lc "cd /workspace/molearn && export PYTHONPATH=/workspace/molearn/src && export MPLBACKEND=Agg && python examples/analysis_SimRNA_RNA.py"
