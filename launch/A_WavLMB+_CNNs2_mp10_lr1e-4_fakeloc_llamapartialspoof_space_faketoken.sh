#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48000
#SBATCH --job-name=weight10_space_faketoken_train_llamapartialspoof_partial #job name
####SBATCH --job-name=space_faketoken_train_partialedit #job name
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --mail-user="lzhan268@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types

#SBATCH -a 2
####SBATCH --dependency=afterok:1491608_0


export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo `date`
source ~/.bashrc
conda activate /home/lzhan268/miniconda3/envs/speechllm

#with fake token with space.
COMMON_ARGS="
    --encoder microsoft/wavlm-base-plus \
    --connector cnn \
    --llm TinyLlama-1.1B-Chat-v1.0 \
    --batch-size 1 \
    --lr 0.0001 \
    --connector-k 2 \
    --meanpool 10 \
    --ft-encoder \
    --group llamapartialspoof_Loc \
    --use-config llamapartialspoof_avdeepfake1m_fakeloc_space_partial.json \
    --total-training-epoch 300 \
    --max-audio-duration 300 \
    --nickname _llamapartialspoof_partial_space_faketoken \
    --add-special-tokens [fake]
"


#with weight on fake token
COMMON_ARGS="
    --encoder microsoft/wavlm-base-plus \
    --connector cnn \
    --llm TinyLlama-1.1B-Chat-v1.0 \
    --batch-size 1 \
    --lr 0.0001 \
    --connector-k 2 \
    --meanpool 10 \
    --ft-encoder \
    --group llamapartialspoof_Loc \
    --use-config llamapartialspoof_avdeepfake1m_fakeloc_space_partial.json \
    --total-training-epoch 300 \
    --max-audio-duration 300 \
    --nickname _llamapartialspoof_space_faketoken \
    --add-special-tokens [fake] \
    --fake-token-loss-weight 10
"

case $SLURM_ARRAY_TASK_ID in
    0)  # 使用文本，概率1.0
        python3 train.py $COMMON_ARGS --use-text --prob-text 1.0 --no-audio
        ;;
    1)  # 使用LoRA
        python3 train.py $COMMON_ARGS --lora-r 8 --lora-alpha 16
        ;;
    2)  # 使用文本 + concat对齐
        python3 train.py $COMMON_ARGS --use-text --prob-text 1.0 --audio-transcript-align concat
        ;;
    3)  # 使用文本 + attention对齐
        python3 train.py $COMMON_ARGS --use-text --prob-text 1.0 --audio-transcript-align attention --num-heads 4
        ;;
    *)
        echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac
