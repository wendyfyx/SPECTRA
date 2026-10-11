#!/bin/bash
#$ -S /bin/bash
#$ -j y -o /path/to/logs
#$ -q compute9.q
#$ -t 1-14400
#$ -tc 400
#$ -N spectra_synth

# ----------------------------------------------------------------------
# Step 2a: synthetic effect experiments (SGE job array, one run per task).
#
# Grid: 3 effect sizes x 8 sample sizes x 3 numbers of affected bundles
#       x 2 localities x 100 repetitions = 14400 tasks.
#
# Set MODE below (2d for the main text, 1d for the supplement), then qsub.
# Each task fits the models once, runs BH-FDR + hFDR at the first alpha_cell,
# then reuses the cached fits for the remaining alpha_cell values.
#
# When all tasks finish, run 02b_synth_aggregate.py once per alpha_cell.
# ----------------------------------------------------------------------

# Maps a 0-based linear job index to per-dimension indices.
#   $1 = linear index, $2..$N = size of each dimension (fastest-varying first)
# Sets the global array IDX[].
qsub_linear_index () {
    local idx=$1
    shift
    IDX=()
    for dim_size in "$@"; do
        IDX+=( $(( idx % dim_size )) )
        idx=$(( idx / dim_size ))
    done
}

# -----------------------------------------
# JOB SETTINGS
# -----------------------------------------
EFFECT_SIZES=(0.2 0.5 0.8)
SAMPLE_SIZES=(50 100 200 400 600 800 1000 1200)
N_AFFECTED_BUNDLES=(5 10 15)
LOCALITIES=(diffuse localized)
REPS=100

MODE=2d
ALPHA_CELLS=(0.05 0.15 0.25)

jobid=$((SGE_TASK_ID - 1))
n1=${#EFFECT_SIZES[@]}
n2=${#SAMPLE_SIZES[@]}
n3=${#N_AFFECTED_BUNDLES[@]}
n4=${#LOCALITIES[@]}

qsub_linear_index "$jobid" "$n1" "$n2" "$n3" "$n4" "$REPS"
effect_size=${EFFECT_SIZES[${IDX[0]}]}
sample_size=${SAMPLE_SIZES[${IDX[1]}]}
n_affected=${N_AFFECTED_BUNDLES[${IDX[2]}]}
locality=${LOCALITIES[${IDX[3]}]}
rep_id=${IDX[4]}

echo "SETTINGS: mode=${MODE} d=${effect_size} n=${sample_size} b=${n_affected} loc=${locality} run=${rep_id} alpha_cells=${ALPHA_CELLS[*]}"

# INPUT / OUTPUT
# (profiles, metadata and bundle list are set at the top of 02a_synth_eval.py)
metric=fa
combination=simes
dir_out_base=/path/to/synthetic_eval/${metric}_${MODE}
dir_out=${dir_out_base}/effect_${effect_size}/sample_${sample_size}/bundles_${n_affected}/${locality}
mkdir -p ${dir_out}

# MAIN
conda activate /path/to/envs/spectra

# No -f: the first call fits and caches the models, later calls reuse the cache
for i in "${!ALPHA_CELLS[@]}"; do
    if [ "$i" -eq 0 ]; then methods=both; else methods=hfdr; fi
    python /path/to/SPECTRA/paper/02a_synth_eval.py \
        -seed 2026 \
        -id   ${rep_id} \
        -o    ${dir_out} \
        -metric ${metric} \
        -mode   ${MODE} \
        -n  ${sample_size} \
        -d  ${effect_size} \
        -b  ${n_affected} \
        -loc ${locality} \
        -ac  ${ALPHA_CELLS[$i]} \
        --methods ${methods} \
        -comb ${combination}
done
