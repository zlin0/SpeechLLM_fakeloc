#!/bin/bash

#SBATCH --job-name=prepare_pe
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --partition=cpu   #queue
#SBATCH --mail-user="lzhan268@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --array=2
####SBATCH --dependency=afterok:1487110_0  # example dependency

source ~/.bashrc
conda activate wedefense_new
DSETs=(train dev eval)
dset=${DSETs[$SLURM_ARRAY_TASK_ID]}

set -x
data_dir="/export/fs05/lzhan268/workspace/wedefense/egs/localization/partialeditv2/v15_ssl_mhfa/data/partialeditv2/E1E2_vctk/"
wav_scp=${data_dir}/${dset}/wav.scp
txt_scp=${data_dir}/${dset}/txt.scp
metadata_file="/export/fs05/arts/dataset/PartialEdit_v2/PartialEdit_E1E2_v2.csv"
save_dir=/export/fs05/lzhan268/workspace/03llm/SpeechLLM_fakeloc/data/partialeditv2/E1E2_vctk

python ./make_partialedit_det_localization.py \
    --wav_scp ${wav_scp} \
    --txt_scp ${txt_scp} \
    --metadata_file ${metadata_file} \
    --save_dir ${save_dir}/${dset}
