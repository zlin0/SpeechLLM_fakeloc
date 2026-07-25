#!/bin/bash

#SBATCH --job-name=prepare
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --partition=cpu   #queue
#SBATCH --mail-user="lzhan268@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --array=0
#SBATCH --dependency=afterok:1491264_0
####SBATCH --dependency=afterok:1487110_0  # example dependency

source ~/.bashrc
conda activate wedefense_new
DSETs=(train val)
dset=${DSETs[$SLURM_ARRAY_TASK_ID]}

# set -x
# data="/export/fs05/lzhan268/workspace/wedefense/egs/localization/avdeepfake1m/v15_ssl_mhfa/data/"
# json_scp=${data}"/avdeepfake1m/"${dset}"_audioonly_realvideo/metadata.scp"
# save_dir=/export/fs05/lzhan268/workspace/03llm/SpeechLLM_fakeloc/data/av-deepfake1m-audioonly_realvideo

# #json_scp=/export/fs05/lzhan268/workspace/wedefense/egs/localization/avdeepfake1m/v15_ssl_mhfa/data/avdeepfake1m/train_audioonly_realvideo/metadata.scp
# #save_dir=/home/lzhan268/workspace/03llm/SpeechLLM_fakeloc/data/av-deepfake1m-audioonly_realvideo

# #python ./local/data_csv_fakeasr/make_avdeepfake1m_det_localization.py \
# python ./make_avdeepfake1m_det_localization.py \
#     --json_scp ${json_scp} \
#     --save_dir ${save_dir} 



set -x
data="/export/fs05/lzhan268/workspace/wedefense/egs/localization/avdeepfake1m++/v15_ssl_mhfa/data/"

save_dir=/export/fs05/lzhan268/workspace/03llm/SpeechLLM_fakeloc/data/avdeepfake1mPP_nosil-audioonly_realvideo

#python ./local/data_csv_fakeasr/make_avdeepfake1m_det_localization.py \
if [ $dset == 'val' ]; then
    # some transcripts are missing...
    json_scp=${data}"/avdeepfake1mPP_nosil/"${dset}"_audioonly_realvideo/metadata_withtranscript.scp"
    save_dir=/export/fs05/lzhan268/workspace/03llm/SpeechLLM_fakeloc/data/avdeepfake1mPP_nosil-audioonly_realvideo_withtranscipt
else
    json_scp=${data}"/avdeepfake1mPP_nosil/"${dset}"_audioonly_realvideo/metadata.scp"
fi
    python ./make_avdeepfake1m_det_localization.py \
        --json_scp ${json_scp} \
        --save_dir ${save_dir}/${dset} 
