#!/bin/bash

MODEL_LIST=(
    "A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit"
    "T_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit"
    "AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0-aliconcat-h0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit"
    "AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0-aliconcat-h0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit_new"  
    "A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc_new" 
    "T_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_avdeepfake1m"
    "AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_avdeepfake1m"
    "A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc"
)

MODEL_LIST=(
    "A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit"
    "T_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit"
    "AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0-aliconcat-h0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_partialedit"
)

JOB_NAME_LIST=(
inf_A_PE
inf_T_PE
inf_AT_PE
inf_AT_PE_new
inf_A_AV1M
inf_T_AV1M
inf_AT_AV1M
inf_A_AV1M_old
)

array=2-5
# idx=0-2
for idx in ${!MODEL_LIST[@]}; do
    model=${MODEL_LIST[$idx]}
    job_name=${JOB_NAME_LIST[$idx]}
    cat launch/zlin/is2026/launch_test.sh | sed -e "s/MODEL_NAME/${model}/g"  |\
     sed -e "s/JOB_NAME/${job_name}/g" | sed -e "s/ARRAY_JOB/${array}/g" |\
       sbatch
done

# inf_AT_AV1M

# check running status in slurm.
# for x in inf_A_PE inf_T_PE inf_AT_PE inf_A_AV1M inf_T_AV1M inf_AT_AV1M inf_A_AV1M_old; do echo =====${x}; showjob |grep ${x};   grep -nRE "Testing|wer_relaxed_mean" ./logs/inference.*${x}*out; done

# for x in inf_A_PE_on_LPS inf_T_PE_on_LPS inf_AT_PE_on_LPS; do echo =====${x}; showjob |grep ${x};   grep -nRE "Testing|wer_relaxed_mean" ./logs/inference.*${x}*out; done