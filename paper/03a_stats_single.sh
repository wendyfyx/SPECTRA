#!/bin/bash
#$ -S /bin/bash
#$ -j y -o /path/to/logs
#$ -q compute9.q
#$ -t 1-504
#$ -tc 200
#$ -N spectra_lmm_dx

# ----------------------------------------------------------------------
# Step 3a: fit the model at every cell of one bundle (SGE job array).
#
# One task per mode x metric x bundle: 2 modes * 4 metrics * 63 bundles = 504.
# Model: metric ~ dx + age + sex + (1 | protocol)
#
# Run once per contrast: target_case=Dementia, then target_case=MCI.
# When all tasks finish, run 03b_stats_aggregate.sh.
# ----------------------------------------------------------------------

bundle_list=/path/to/metadata/bundle_list_n63.txt
metadata_path=/path/to/metadata/metadata.csv    # columns: subject, dx, age, sex, protocol

MODES=(1d 2d)
METRICS=(fa md rd axd)
BUNDLES=(`cat ${bundle_list}`)

jobid=$((${SGE_TASK_ID}-1))
nb=${#BUNDLES[@]}
nm=${#METRICS[@]}

mode_idx=$((jobid / (nm * nb)))
within=$((jobid % (nm * nb)))
metric_idx=$((within / nb))
bundle_idx=$((within % nb))

mode=${MODES[$mode_idx]}
metric=${METRICS[$metric_idx]}
bundle=${BUNDLES[$bundle_idx]}

s_len=5
r_len=5
target_var=dx
target_ref=CN
target_case=Dementia

profile_dir=/path/to/profiles/s${s_len}_r${r_len}
output_dir=/path/to/stats/${metric}__s${s_len}_r${r_len}_${mode}__${target_var}${target_case}
mkdir -p ${output_dir}

conda activate /path/to/envs/spectra

spectra_lmm_fdr single \
    -meta ${metadata_path} \
    -root ${profile_dir} \
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
    -stats_mode hfdr_resel \
    -bundle ${bundle} \
    -q
