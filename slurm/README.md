# Molearn container on a Slurm cluster

Slurm stays on the host cluster. The container supplies Molearn and its Python,
CUDA-runtime, OpenMM, and analysis dependencies. Most HPC systems run Docker
images through Apptainer/Singularity because Docker daemons are not permitted
on compute nodes.

## Build and publish the Docker image

From the repository root on a machine with Docker:

```bash
docker build -t YOUR_REGISTRY/molearn-rna:latest .
docker push YOUR_REGISTRY/molearn-rna:latest
```

Do not put trajectories in the image. `.dockerignore` excludes the large TRC,
NetCDF, checkpoint, and generated-analysis files.

## Convert it on the cluster

```bash
module load apptainer
apptainer pull molearn-rna.sif docker://YOUR_REGISTRY/molearn-rna:latest
```

For a private registry, authenticate using the mechanism recommended by the
cluster administrators before pulling.

## Verify GPU access

```bash
apptainer exec --nv molearn-rna.sif python -c \
  "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The host NVIDIA driver must support the CUDA runtime installed in the image.

## Use your existing Modeller license

Keep the key outside both Git and the container image. On the cluster, put only
the key in a user-readable file and protect it:

```bash
chmod 600 /path/to/modeller.key
```

Pass that filename during submission. The job script reads it into Modeller's
supported `KEY_MODELLER` variable inside the container:

```bash
sbatch --export=ALL,MOLEARN_IMAGE="$PWD/molearn-rna.sif",MOLEARN_PROJECT="$PWD",MODELLER_LICENSE_FILE="/path/to/modeller.key" \
  slurm/train_simrna_apptainer.sbatch
```

Do not place the key directly in an `sbatch` command, Dockerfile, repository,
job log, or public/private registry image. If the RNA reconstruction workflow
does not invoke Modeller, `MODELLER_LICENSE_FILE` may be omitted.

## Submit SimRNA training

Run from the repository root on cluster storage:

```bash
sbatch --export=ALL,MOLEARN_IMAGE="$PWD/molearn-rna.sif",MOLEARN_PROJECT="$PWD" \
  slurm/train_simrna_apptainer.sbatch
```

Optional settings can be exported at submission time:

```bash
sbatch --export=ALL,MOLEARN_IMAGE="$PWD/molearn-rna.sif",MOLEARN_PROJECT="$PWD",FRAME_STEP=5,BATCH_SIZE=32 \
  slurm/train_simrna_apptainer.sbatch
```

## Submit SimRNA analysis

```bash
sbatch --export=ALL,MOLEARN_IMAGE="$PWD/molearn-rna.sif",MOLEARN_PROJECT="$PWD" \
  slurm/analyse_simrna_apptainer.sbatch
```

## Submit three-state MD RNA training

```bash
sbatch --export=ALL,MOLEARN_IMAGE="$PWD/molearn-rna.sif",MOLEARN_PROJECT="$PWD" \
  slurm/train_md_rna_apptainer.sbatch
```

Adjust partition/account directives and requested memory/time/GPU resources in
the `.sbatch` files to match the local cluster policy. A Modeller license key
may also need to be configured according to the cluster's licensing setup.
