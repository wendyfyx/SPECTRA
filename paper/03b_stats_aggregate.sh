#!/bin/bash
#$ -S /bin/bash
#$ -j y -o /path/to/logs
#$ -q compute9.q
#$ -t 1-8
#$ -N spectra_lmm_dx_agg

# ----------------------------------------------------------------------
# Step 3b: multiple-comparison correction across all bundles (SGE job array).
#
# One task per mode x metric: 2 modes * 4 metrics = 8.
# Loads the fits saved by 03a_stats_single.sh and applies BH-FDR and
# hFDR-RESEL. Use the same settings (target_case, paths) as in step 3a.
#
# Output per mode and metric:
#   fdr__stats_all.csv, fdr__stats_summary.csv
#   hfdr_resel__stats_blocks.csv
#   hfdr_resel__stats_all_ac{05,15,25}.csv, hfdr_resel__stats_summary_ac{05,15,25}.csv
# ----------------------------------------------------------------------

bundle_list=/path/to/metadata/bundle_list_n63.txt
metadata_path=/path/to/metadata/metadata.csv

MODES=(1d 2d)
METRICS=(fa md rd axd)

jobid=$((${SGE_TASK_ID}-1))
mode=${MODES[$((jobid / 4))]}
metric=${METRICS[$((jobid % 4))]}

s_len=5
r_len=5
target_var=dx
target_ref=CN
target_case=Dementia

profile_dir=/path/to/profiles/s${s_len}_r${r_len}
output_dir=/path/to/stats/${metric}__s${s_len}_r${r_len}_${mode}__${target_var}${target_case}

conda activate /path/to/envs/spectra

base_cmd="spectra_lmm_fdr aggregate \
    -meta ${metadata_path} \
    -root ${profile_dir} \
    -bundle_list ${bundle_list} \
    -metric ${metric} \
    -mode ${mode} \
    -subj subject \
    -target ${target_var} \
    -target_ref ${target_ref} \
    -target_case ${target_case} \
    -group protocol \
    -covar age sex \
    -cont age \
    -o ${output_dir} \
    -s ${s_len} \
    -r ${r_len} \
    -q"

echo "=== Aggregating FDR: ${mode} ${metric} ==="
eval "${base_cmd} -stats_mode fdr -alpha 0.05"

echo "=== Aggregating hFDR-RESEL: ${mode} ${metric} ==="
eval "${base_cmd} -stats_mode hfdr_resel -alpha_block 0.05 -alpha_cell 0.05 0.15 0.25 -combination simes"
