#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
####SBATCH --job-name=inference_A_AV1M_on_PE
#SBATCH --job-name=zero
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu   #queue
###SBATCH --partition=gpu-a100   #queue
###SBATCH --account=a100acct
#SBATCH --mail-user="lzhan268@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --error=logs/inference.%x_%A_%a.err
#SBATCH --output=logs/inference.%x_%A_%a.out
###SBATCH --exclude=c03,c04,c08,c09,c10,c11,c14,c15,c16,c18,c19,c23,e01,e03,octopod
#SBATCH -a 0-2
####SBATCH --nodelist=c19
####SBATCH --dependency=aftercorr:1472466
####SBATCH --dependency=aftercorr:1530055_2

source ~/.bashrc	
echo `date`

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'


model="zero"
test_on="A"

CONFs=(
partialedit_partialedit_fakeloc_dev.json
partialedit_partialedit_fakeloc.json
avdeepfake1m_avdeepfake1m_fakeloc.json
)
conf_name=${CONFs[$SLURM_ARRAY_TASK_ID]}


conda activate /home/lzhan268/miniconda3/envs/speechllm

# zero-shot
test_on="A"
python3 test_before_multisample.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'cnn' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --batch-size 1 \
    --meanpool 100 \
    --use-config ${conf_name} \
    --checkpoints "checkpoints/zero" \
    --epoch-to-test 0   \
    --test-on ${test_on}

