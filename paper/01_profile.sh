#!/bin/bash
#$ -S /bin/bash
#$ -j y -o /path/to/logs
#$ -q compute9.q
#$ -t 1-1209
#$ -tc 200
#$ -N spectra_profile

# ----------------------------------------------------------------------
# Step 1: 1D and 2D tract profiles (SGE job array, one subject per task).
#
# Set -t to 1-<number of subjects>. Run once per cohort (ADNI, HABS-HD).
#
# Input per subject:
#   ${subj_dir}/rec_bundles/<bundle>.trk   bundle in MNI space
#   ${subj_dir}/org_bundles/<bundle>.trk   same bundle in native space
#   ${subj_dir}/dti/<metric>.nii.gz        scalar maps in native space
# Output:
#   ${output_dir}/<metric>/<bundle>/<subject>.npz
# ----------------------------------------------------------------------

SUBJECTS=(`cat /path/to/metadata/subject_list.txt`)
subj=${SUBJECTS[${SGE_TASK_ID}-1]}
echo ${subj}

subj_dir=/path/to/derivatives/${subj}
atlas=/path/to/ENIGMA_tract_atlas/bundles
BUNDLES=(`cat /path/to/ENIGMA_tract_atlas/bundle_list.txt`)

s_len=5
r_len=5
output_dir=/path/to/profiles/s${s_len}_r${r_len}

conda activate /path/to/envs/spectra

for bundle in "${BUNDLES[@]}"; do
    echo -e "\nRUNNING FOR ${bundle}"
    mkdir -p ${output_dir}/fa/${bundle}
    mkdir -p ${output_dir}/md/${bundle}
    mkdir -p ${output_dir}/rd/${bundle}
    mkdir -p ${output_dir}/axd/${bundle}

    if [ -f ${subj_dir}/rec_bundles/${bundle}.trk ]; then
        spectra_profile \
            -rec ${subj_dir}/rec_bundles/${bundle}.trk \
            -org ${subj_dir}/org_bundles/${bundle}.trk \
            -ref ${atlas}/${bundle}.trk \
            -maps ${subj_dir}/dti/fa.nii.gz ${subj_dir}/dti/md.nii.gz ${subj_dir}/dti/rd.nii.gz ${subj_dir}/dti/axd.nii.gz \
            -profiles ${output_dir}/fa/${bundle}/${subj} ${output_dir}/md/${bundle}/${subj} ${output_dir}/rd/${bundle}/${subj} ${output_dir}/axd/${bundle}/${subj} \
            -s ${s_len} \
            -r ${r_len} \
            -min 20 \
            -robust linear \
            -f
    fi
done
