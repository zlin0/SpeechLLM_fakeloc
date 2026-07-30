#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16000
#SBATCH --job-name=inf_lora_ft
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu   #queue
##SBATCH --partition=gpu-a100   #queue
##SBATCH --account=a100acct
#SBATCH --mail-user="lzhan268@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --error=logs/inference.%x_%A_%a.err
#SBATCH --output=logs/inference.%x_%A_%a.out
#SBATCH --exclude=c05
#SBATCH -a 0
####SBATCH --dependency=aftercorr:1529612_0

source ~/.bashrc	
echo `date`

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'


# for comparing lora/finetune.
MODELs=(
    "A_wavlm-base-plus-cnn-TinyLlama-bs1_nolora_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc"
    "A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_mp10_str2_lr0.0001_T_avdeepfake1m"
)
model=${MODELs[$SLURM_ARRAY_TASK_ID]}
conf_name=avdeepfake1m_avdeepfake1m_fakeloc.json


# # For partialedit text-only.
# model="T_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit"
# CONFs=(
# partialedit_partialedit_fakeloc_dev.json
# partialedit_partialedit_fakeloc.json
# avdeepfake1m_avdeepfake1m_fakeloc.json
# )
# conf_name=${CONFs[$SLURM_ARRAY_TASK_ID]}


if [[ $model == AT_* ]]; then
    test_on="AT"
else
    test_on=${model:0:1}  # A/T
fi


conda activate speechllm

# some problem in test.py with resampling implemnet.
python3 test_before_multisample.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'cnn' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --batch-size 1 \
    --meanpool 100 \
    --use-config ${conf_name} \
    --checkpoints "checkpoints/${model}/last.ckpt" \
    --epoch-to-test 0   \
    --test-on ${test_on}


# test_data=${conf_name %.json}
# python ./update_result_files.py \
#     --input checkpoints/${model}/last/${conf_name}/A/${conf_name}_before_multisample.csv \
#     --recalculate

# cat checkpoints/${model}/last/${conf_name}/A/${conf_name}_before_multisample.csv.wer_wfler