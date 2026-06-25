# PNGF Target-Conditioned ES Generator

This path trains a one-shot inverse generator for the PNGF paper antenna:

```text
target [Re(S11), Im(S11), D] at 25, 27.5, 30, 32.5, 35 GHz
  -> CNN generator
  -> hard 21x21 mask
  -> two-axis symmetry + center-feed projection
  -> external PNGF scorer
  -> ES update
```

PNGF is an external dependency at `C:\Users\hczupryna\dev\PNGF`; its GPL/AGPL source is not vendored here.

## Required PNGF State

For the paper center-fed substrate setup, the external PNGF build needs:

```text
C:\Users\hczupryna\dev\PNGF\pngf-opt\Gmat_sub_01.bin
C:\Users\hczupryna\dev\PNGF\pngf-opt\Gmat_sub_02.bin
C:\Users\hczupryna\dev\PNGF\pngf-opt\Gmat_sub_03.bin
C:\Users\hczupryna\dev\PNGF\pngf-opt\Gmat_sub_04.bin
C:\Users\hczupryna\dev\PNGF\pngf-opt\Gmat_sub_05.bin
```

and should be compiled with the center-fed path, not side-launch:

```bash
make clean
make DEFINES="-DUSE_SIDE_LAUNCH_FEED=0"
```

At the time this integration was added, those `Gmat_sub_*.bin` files were not present locally, so real uncached PNGF scoring cannot run yet.

## Dataset Export

After running PNGF-DBS trajectories, export accepted and rejected flip candidates:

```bash
python export_pngf_dbs_dataset.py \
  --logs "C:\Users\hczupryna\dev\PNGF\pngf-opt\log_*.txt" \
  --output-npz eggroll_runs/pngf_dbs_dataset.npz
```

The output contains:

```text
masks[N,21,21]
targets[N,15]
objective[N]
accepted[N]
seed[N]
run_id[N]
step[N]
flip_index[N]
freq_hz[5]
```

## External Scorer Contract

`PNGFScorer` writes an input `.npz`:

```text
masks[N,21,21]        uint8, already projected
freq_hz[5]
solver_mode
geometry
```

The scorer command must write an output `.npz` with either:

```text
targets[N,15]
valid[N]
error_message[N]
```

or:

```text
s11_re[N,5]
s11_im[N,5]
directivity[N,5]
valid[N]
error_message[N]
```

Failed rows are mapped to a large bad target value instead of crashing the ES loop.

## Training

Run supervised reconstruction pretraining plus ES fine-tuning:

```bash
python train_pngf_es_generator.py \
  --dataset-npz eggroll_runs/pngf_dbs_dataset.npz \
  --mode both \
  --pngf-work-dir "C:\Users\hczupryna\dev\PNGF" \
  --pngf-command "python path\to\pngf_batch_score.py --input \"{input_npz}\" --output \"{output_npz}\"" \
  --population-size 8 \
  --steps 10 \
  --rank 1 \
  --sigma 0.10
```

Outputs include CSV logs, checkpoint `.npz` files, and `best_pngf_design.png`.

## Evaluation

Compare pretrained and ES-finetuned checkpoints on the same PNGF target set:

```bash
python evaluate_pngf_generator.py \
  --dataset-npz eggroll_runs/pngf_dbs_dataset.npz \
  --checkpoints \
    eggroll_runs/pngf_es/pngf_generator_pretrained.npz \
    eggroll_runs/pngf_es/pngf_generator_final.npz \
  --random-baseline 3 \
  --pngf-work-dir "C:\Users\hczupryna\dev\PNGF" \
  --pngf-command "python path\to\pngf_batch_score.py --input \"{input_npz}\" --output \"{output_npz}\"" \
  --output-csv eggroll_runs/pngf_eval.csv
```

This scores hard projected masks through PNGF and reports target error, S11 error, directivity error, mask uniqueness, fill ratio, and valid solve ratio.
