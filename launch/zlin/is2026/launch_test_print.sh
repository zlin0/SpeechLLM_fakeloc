#!/bin/bash

source ~/.bashrc	
echo `date`

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'


MODELs=(
A_wavlm-base-plus-cnn-TinyLlama-bs1_nolora_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc_old
)


MODELs=(
    "A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit"
    "T_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit"
    "AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0-aliconcat-h0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit" 
    "AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0-aliconcat-h0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit_new" 
    "A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc"   
    "A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc_new" 
    "T_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_avdeepfake1m"
    "AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_avdeepfake1m"
)

# #LORA
# MODELs=(
#     A_wavlm-base-plus-cnn-TinyLlama-bs1_nolora_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc
#     A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_mp10_str2_lr0.0001_T_avdeepfake1m
# )


CONFs=(
partialedit_partialedit_fakeloc_dev.json
partialedit_partialedit_fakeloc.json
avdeepfake1m_avdeepfake1m_fakeloc.json
)

TEST_DATAs=(
    partialedit_dev  
    partialedit_eval
    av-deepfake1m_val
)


# A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit --- partialedit_partialedit_fakeloc_dev.json --- partialedit_partialedit_fakeloc_dev --- A
# cat: checkpoints/A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit/last/partialedit_partialedit_fakeloc_dev/A/partialedit_partialedit_fakeloc_dev_before_multisample.csv.wer_wfler: No such file or directory

conda activate speechllm

# for idx in ${!MODELs[@]}; do
for model in ${MODELs[@]}; do

    for idx in ${!CONFs[@]}; do
        conf_name=${CONFs[$idx]}
        test_data=${TEST_DATAs[$idx]}

        if [[ $model == AT_* ]]; then
            test_mode="AT"
        else
            test_mode=${model:0:1}  # A/T
        fi

        echo "${model} --- ${conf_name} --- ${test_data} --- ${test_mode}"
        wc -l data/${test_data}.csv
        wc -l checkpoints/${model}/last/${test_data}/${test_mode}/${test_data}_before_multisample.csv
        
        cat checkpoints/${model}/last/${test_data}/${test_mode}/${test_data}_before_multisample.csv.wer_wfler
        python ./update_result_files.py \
            --input checkpoints/${model}/last/${test_data}/${test_mode}/${test_data}_before_multisample.csv \
            --recalculate

        
    done
done

# checkpoints/A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc_new/last/partialedit_dev/A