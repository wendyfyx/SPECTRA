#!/bin/bash

# ----------------------------------------------------------------------
# Step 3c: effect-size maps on the atlas bundles.
#
# Run locally (not a qsub script): bash 03c_stats_maps.sh
# Rendering needs a display or offscreen OpenGL.
#
# For every mode and metric:
#   1. spectra_export_stats writes one Cohen's d grid per bundle, keeping
#      only the cells significant under hFDR-RESEL (alpha_cell = 0.15).
#   2. spectra_viz colors the atlas bundles by those grids.
# ----------------------------------------------------------------------

slen=5
rlen=5
test=dxDementia
stat_mode=hfdr_resel
suffix=_ac15
metrics=(fa md rd axd)
modes=(1d 2d)

dir_atlas=/path/to/ENIGMA_tract_atlas/bundles_downsampled
dir_stats_root=/path/to/stats
bundle_list=/path/to/metadata/bundle_list_n63.txt

stat=d_cohen

# 1. Export the significant Cohen's d grids
for mode in "${modes[@]}"; do
for metric in "${metrics[@]}"; do
    dir_test=${dir_stats_root}/${metric}__s${slen}_r${rlen}_${mode}__${test}
    spectra_export_stats \
        -i ${dir_test}/${stat_mode}__stats_all${suffix}.csv \
        -o ${dir_test}/grids/${stat_mode}${suffix}/ \
        -export_stats ${stat} \
        -sig_only \
        -bundle_list ${bundle_list}
done
done

# 2. Render one figure per mode and metric
# USAGE: plot_bundles <name> <view> <bundle> [<bundle> ...]
#   One bundle renders it alone; several bundles are rendered together.
plot_bundles() {
    local name=$1
    local view=$2
    shift 2
    local bundles=("$@")

    for mode in "${modes[@]}"; do
    for metric in "${metrics[@]}"; do
        dir_test=${dir_stats_root}/${metric}__s${slen}_r${rlen}_${mode}__${test}
        dir_grids=${dir_test}/grids/${stat_mode}${suffix}
        dir_plots=${dir_test}/bundle_viz/${stat_mode}${suffix}
        mkdir -p ${dir_plots}

        input_tracts=$(printf "${dir_atlas}/%s.trk " "${bundles[@]}")
        input_stats=$(printf "${dir_grids}/%s__${stat}.npy " "${bundles[@]}")

        spectra_viz \
            -i ${input_tracts} \
            -glass default_glass \
            -c ${input_stats} \
            -o ${dir_plots}/${metric}_${stat}__${name}__${view}.png \
            -cam ${view} \
            -crop \
            -s ${slen} \
            -r ${rlen} \
            -robust linear \
            -cmap coolwarm \
            -vmin -0.8 \
            -vmax 0.8 \
            -ctitle ${stat} \
            -cbar ${dir_plots}/${metric}_${stat}_cbar.png
    done
    done
}

# Association
plot_bundles ILF_L  Sagittal_L ILF_L
plot_bundles ILF_R  Sagittal_R ILF_R
plot_bundles IFOF_L Sagittal_L IFOF_L
plot_bundles IFOF_R Sagittal_R IFOF_R

# Limbic
plot_bundles CG_L Sagittal_L CG_FP_L CG_PH_L
plot_bundles CG_R Sagittal_R CG_FP_R CG_PH_R

# Projection
plot_bundles CPT_F_L Sagittal_L CPT_F_L
plot_bundles CPT_F_R Sagittal_R CPT_F_R
plot_bundles CPT_P_L Sagittal_L CPT_P_L
plot_bundles CPT_P_R Sagittal_R CPT_P_R
plot_bundles CS_L Sagittal_L CS_A_L CS_Pre_L CS_Post_L
plot_bundles CS_R Sagittal_R CS_A_R CS_Pre_R CS_Post_R

# Commissural
plot_bundles Commissural Sagittal_L CC_ForcepsMajor CC_ForcepsMinor CC_Body_Pre CC_Body_Post CC_Tapetum
plot_bundles Commissural Sagittal_R CC_ForcepsMajor CC_ForcepsMinor CC_Body_Pre CC_Body_Post CC_Tapetum
