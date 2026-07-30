#!/bin/bash
#SBATCH --ntasks-per-node=1
####SBATCH --cpus-per-task=8
#SBATCH --mem=16000
####SBATCH --job-name=inference_PE_on_LPS
#SBATCH --job-name=JOB_NAME_on_LPS
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu   #queue
###SBATCH --partition=gpu-a100   #queue
###SBATCH --account=a100acct
#SBATCH --mail-user="lzhan268@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --error=logs/inference.%x_%A_%a.err
#SBATCH --output=logs/inference.%x_%A_%a.out
#SBATCH --exclude=c05
###SBATCH --exclude=c03,c04,c08,c09,c10,c11,c14,c15,c16,c18,c19,c23,e01,e03,octopod
#SBATCH -a ARRAY_JOB
####SBATCH --nodelist=c19
####SBATCH --dependency=aftercorr:1472466
####SBATCH --dependency=aftercorr:1530055_2

source ~/.bashrc	
echo `date`

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'



# # old version
# MODELs=(
# AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p0.5_ft_encoder-lr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m
# A_wavlm-base-plus-cnn-TinyLlama-bs1_ft_encoder_mp10_str2_lr0.0001_T_avdeepfake1m
# A_wavlm-base-plus-cnn-TinyLlama-bs1_ft_encoder_mp10_str2_lr0.0001_T_avdeepfake1m_asr
# A_wavlm-base-plus-cnn-TinyLlama-bs1_lora16-16_ft_encoder-lr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
# A_wavlm-base-plus-cnn-TinyLlama-bs1_lora32-16_ft_encoder-lr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
# A_wavlm-base-plus-cnn-TinyLlama-bs1_lora4-16_ft_encoder-lr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
# A_wavlm-base-plus-cnn-TinyLlama-bs1_lora64-16_ft_encoder-lr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
# A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16-lr0.0002_mp10_str2_lr0.01_T_avdeepfake1m
# A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16-lr2e-08_mp10_str2_lr1e-06_T_avdeepfake1m
# A_wavlm-base-plus-cnn-TinyLlama-bs1_mp10_str2_lr0.0001_T_avdeepfake1m
# A_wavlm-base-plus-cnn-TinyLlama-bs1_nolora_ft_encoder-lr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
# )

MODELs=(
A_wavlm-base-plus-cnn-TinyLlama-bs1_nolora_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc_old
)


#MODELs=(
#A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
#A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-4_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
#A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
#A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-64_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
#A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-128_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
#)
# A_wavlm-base-plus-cnn-TinyLlama-bs1_nolora_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
# model=${MODELs[$SLURM_ARRAY_TASK_ID]} 


model="A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit"
model="AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0-aliconcat-h0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit"
model="A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc_new"
model="AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_avdeepfake1m"
model="MODEL_NAME"

if [[ $model == AT_* ]]; then
    test_on="AT"
else
    test_on=${model:0:1}  # A/T
fi

CONFs=(
partialedit_partialedit_fakeloc_dev.json
partialedit_partialedit_fakeloc.json
avdeepfake1m_avdeepfake1m_fakeloc.json
)
conf_name=${CONFs[$SLURM_ARRAY_TASK_ID]}


CONFs=(
dev01-yourtts-partial-cf
dev01-cosyvoice-partial-cf
dev01-elevenlab-partial-cf
dev01-gptsovits-partial-cf
dev01-ljjets-partial-cf
dev01-xttsv2-partial-cf
)
conf_name=avdeepfake1m_llamapartialspoof_${CONFs[$SLURM_ARRAY_TASK_ID]}_fakeloc.json


conda activate /home/lzhan268/miniconda3/envs/speechllm

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



# python3 test.py \
#     --encoder 'microsoft/wavlm-base-plus' \
#     --connector 'cnn' \
#     --llm 'TinyLlama-1.1B-Chat-v1.0' \
#     --batch-size 1 \
#     --data-dir llamapartialspoof \
#     --checkpoint "checkpoints/${model}/last.ckpt"

#TODO
# mv checkpoints/${model}_fakeloc/last/llamapartialspoof/A/llamapartialspoof.txt checkpoints/${model}_fakeloc/last/llamapartialspoof/A/llamapartialspoof.txt.old

# python3 test.py \
#     --encoder 'microsoft/wavlm-base-plus' \
#     --connector 'cnn' \
#     --llm 'TinyLlama-1.1B-Chat-v1.0' \
#     --batch-size 1 \
#     --meanpool 100 \
#     --use-config ${conf_name} \
#     --checkpoints "checkpoints/${model}/last.ckpt" \
#     --epoch-to-test 0
    # --test_on A/AT/T
    # --epoch-to-test 15

