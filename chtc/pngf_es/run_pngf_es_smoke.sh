#!/usr/bin/env bash
set -euo pipefail

CLUSTER_ID="${1:-local}"
PROCESS_ID="${2:-0}"
RESULT_ARCHIVE="results_${CLUSTER_ID}_${PROCESS_ID}.tar.gz"

finish() {
  status=$?
  mkdir -p job_diagnostics
  printf "%s\n" "$status" > job_diagnostics/exit_status.txt
  {
    echo "status=$status"
    echo "date=$(date -Is)"
    echo "pwd=$PWD"
    find results -maxdepth 3 -type f -printf "%p %s\n" 2>/dev/null || true
  } > job_diagnostics/manifest.txt
  tar -czf "$RESULT_ARCHIVE" job_diagnostics results 2>/dev/null || tar -czf "$RESULT_ARCHIVE" job_diagnostics
  ls -lh "$RESULT_ARCHIVE"
  exit "$status"
}
trap finish EXIT

echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo "Cluster/Process: ${CLUSTER_ID}/${PROCESS_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python3}")"
PYTHON_BIN="$(readlink -f "${PYTHON_BIN}" 2>/dev/null || printf "%s" "${PYTHON_BIN}")"
export PYTHON_BIN
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" - <<'PY'
import sys
print("sys.executable", repr(sys.executable))
PY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-12}"
export GOTO_NUM_THREADS="${GOTO_NUM_THREADS:-12}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"

mkdir -p results
tar -xzf pixelant_es_code.tar.gz

PYTHON_DEPS="${PWD}/python_deps"
PIP_CACHE_DIR="${PWD}/.pip_cache"
TMPDIR="${PWD}/tmp"
HYPERSCALEES_DIR="${PWD}/HyperscaleES"
mkdir -p "${PYTHON_DEPS}" "${PIP_CACHE_DIR}" "${TMPDIR}"
export PYTHONPATH="${PYTHON_DEPS}:${PWD}:${PYTHONPATH:-}"
export PIP_CACHE_DIR TMPDIR

"${PYTHON_BIN}" -m pip install --no-cache-dir --only-binary=:all: --target "${PYTHON_DEPS}" \
  numpy==1.26.4 \
  scipy==1.13.1 \
  matplotlib==3.8.4 \
  optax==0.2.2 \
  jax==0.4.30 \
  jaxlib==0.4.30

git clone https://github.com/ESHyperscale/HyperscaleES.git "${HYPERSCALEES_DIR}"
git -C "${HYPERSCALEES_DIR}" checkout b77f7d6f91238fd575313e946b9cad21e0a74b32
cat > "${HYPERSCALEES_DIR}/src/hyperscalees/__init__.py" <<'PY'
from . import models, noiser
__version__ = "0.0.1-pixelant-runtime"
PY
cat > "${HYPERSCALEES_DIR}/src/hyperscalees/models/__init__.py" <<'PY'
from . import common, base_model
PY
export PYTHONPATH="${HYPERSCALEES_DIR}/src:${PYTHONPATH}"

"${PYTHON_BIN}" - <<'PY'
import hyperscalees
import jax
import numpy
import optax
print("Python dependencies import successfully")
print("hyperscalees", getattr(hyperscalees, "__version__", "source"))
print("jax", jax.__version__, "numpy", numpy.__version__)
PY

chmod +x evaluate-fixed-design evaluate-fixed-design-batch || true
for matrix in Gmat_sub_01.bin Gmat_sub_02.bin Gmat_sub_03.bin Gmat_sub_04.bin Gmat_sub_05.bin; do
  test -s "$matrix"
  ls -lh "$matrix"
done

SCORER_CMD="${PYTHON_BIN} ${PWD}/pngf_npz_batch_scorer.py --input \"{input_npz}\" --output \"{output_npz}\" --evaluator ${PWD}/evaluate-fixed-design-batch --matrix-dir ${PWD}"
echo "Scorer command: ${SCORER_CMD}"

"${PYTHON_BIN}" -u make_pngf_smoke_dataset.py \
  --output results/pngf_smoke_dataset.npz \
  --scorer-command "${SCORER_CMD}" \
  --count 4 \
  --seed 20260616 \
  --cache-dir results/pngf_cache

"${PYTHON_BIN}" -u train_pngf_es_generator.py \
  --dataset-npz results/pngf_smoke_dataset.npz \
  --output-dir results/train \
  --mode both \
  --pretrain-epochs 2 \
  --pretrain-batch-size 4 \
  --population-size 2 \
  --steps 2 \
  --channels 4 \
  --rank 1 \
  --sigma 0.10 \
  --lr 1e-3 \
  --save-every 1 \
  --pngf-command "${SCORER_CMD}" \
  --pngf-work-dir "${PWD}" \
  --pngf-cache-dir results/pngf_cache

"${PYTHON_BIN}" -u evaluate_pngf_generator.py \
  --dataset-npz results/pngf_smoke_dataset.npz \
  --checkpoints results/train/pngf_generator_pretrained.npz results/train/pngf_generator_final.npz \
  --random-baseline 1 \
  --limit 4 \
  --output-csv results/pngf_eval.csv \
  --pngf-command "${SCORER_CMD}" \
  --pngf-work-dir "${PWD}" \
  --pngf-cache-dir results/pngf_cache

echo "PNGF_ES_SMOKE_DONE"
