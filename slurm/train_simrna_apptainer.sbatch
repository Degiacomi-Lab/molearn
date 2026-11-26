#!/bin/bash
#SBATCH --job-name=molearn-simrna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Override when submitting if your cluster uses different locations:
# sbatch --export=ALL,MOLEARN_IMAGE=/path/molearn-rna.sif,MOLEARN_PROJECT=/path/molearn ...
MOLEARN_IMAGE="${MOLEARN_IMAGE:?Set MOLEARN_IMAGE to the .sif image path}"
MOLEARN_PROJECT="${MOLEARN_PROJECT:-${SLURM_SUBMIT_DIR}}"
FRAME_STEP="${FRAME_STEP:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"

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
    bash -lc "cd /workspace/molearn && export PYTHONPATH=/workspace/molearn/src && python examples/simrna_foldingnet_basic_RNA.py --ensembles apo holo --frame-step ${FRAME_STEP} --batch-size ${BATCH_SIZE}"
