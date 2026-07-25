from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
# from trainer import SpeechLLMLightning
from trainer_0206_before_multiple_sample import SpeechLLMLightning
from trainer_multitask import SpeechLLMMultitaskLightning
from dataset import InstructionalAudioDataset

import torch.utils.data as data_utils
from dataset import InstructionalAudioDataset, MyCollator, CompositeAudioDataset
from utils import get_model_config 
import os
import shutil
import logging


def main(model_config):
    print(model_config)
    trainer_variant = model_config.get('trainer_variant', 'baseline')
    model_cls = SpeechLLMLightning
    binary_classification = False
    if trainer_variant == 'multitask':
        model_cls = SpeechLLMMultitaskLightning
        binary_classification = True

    # model = SpeechLLMLightning.load_from_checkpoint(f"checkpoints/{model_name}/last.ckpt")
    import os
    if model_config['checkpoints'] and os.path.isfile(model_config['checkpoints']):
        model = model_cls.load_from_checkpoint(
            model_config['checkpoints'],
            **model_config,
        )
        # Ensure test-time sampling params follow CLI/config
        for key in [
            "test_use_batch_sampling",
            "test_num_samples",
            "test_max_new_tokens",
            "test_do_sample",
            "test_top_p",
            "test_top_k",
            "test_temperature",
            "icl_num_shots",
            "icl_train_csv",
            "icl_seed",
        ]:
            if key in model_config:
                setattr(model, key, model_config[key])
    # elif model_config['epoch_to_test'] > 0: #TODO check
    #     version = f"{model_config['model_name']}epoch-epoch={model_config['epoch_to_test']}"
    #     i=0
    #     while os.path.exists(f"checkpoints/{model_config['model_name']}/{version}-v{i+1}.ckpt"):
    #         i+=1
    #     print(f"Using epoch {model_config['epoch_to_test']} version {i}")
    #     if i==0: model = SpeechLLMLightning.load_from_checkpoint(f"checkpoints/{model_config['model_name']}/{version}.ckpt")
        # else:    model = SpeechLLMLightning.load_from_checkpoint(f"checkpoints/{model_config['model_name']}/{version}-v{i}.ckpt")
    else:  
        print("Using untrained model!")
        version='base'
        model = model_cls(**model_config)
    tokenizer = model.llm_tokenizer
    trainer = Trainer(
        accelerator='gpu', devices=1, log_every_n_steps=100, callbacks=[TQDMProgressBar(refresh_rate=50)]
    )
    print("Model loaded")

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if model_config['test_on']=='AT': text_options = [("AT", True, True), ("A", False, True), ("T", True, False)]
    # if model_config['test_on']=='AT': text_options = [("AT", True, True)]
    # if model_config['test_on']=='AT': text_options = [ ("A", False, True), ("T", True, False)]
    elif model_config['test_on']=='T': text_options = [('T', True, False)]
    elif model_config['test_on']=='A': text_options = [('A', False, True)]
    # text_options = [('A', False, True)]
    # text_options = [("AT", True, True), ("T", True, False), ("A", False, True)]

    for test_set in model_config['test_sets']:
        for suffix, use_text, use_audio in text_options:
            
            ckpt_path = model_config['checkpoints']
            if ckpt_path.endswith('.ckpt'):
                ckpt_path = ckpt_path[:-5]
            log_dir = os.path.join(ckpt_path, test_set, suffix)
            try:
                os.makedirs(log_dir)
            except:
                print(f"{log_dir} exists")
            #update logger
            logger.handlers.clear()
            file_handler = logging.FileHandler(os.path.join(log_dir, f"{test_set}.txt"), mode="w")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            print(f"Testing {test_set} using {suffix}")

            test_dataset = InstructionalAudioDataset(
                csv_file=f'./data/{test_set}.csv',
                mode='test', 
                max_len=model_config['max_number_seconds'],
                fields=model_config['test_sets'][test_set],
                use_text=use_text,
                prob_text=1,
                audio_transcript_align=model_config.get('audio_transcript_align', None),
                use_audio=use_audio,
                frame_binary_classification=binary_classification,
                icl_num_shots=model_config.get('icl_num_shots', 0),
                icl_train_csv=model_config.get('icl_train_csv', None),
                icl_seed=model_config.get('icl_seed', 42),
                test_on=suffix,
            )
                            # audio_transcript_align = model_config.get('audio_transcript_align', None)
            my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer, use_text=use_text)
            test_loader = data_utils.DataLoader(test_dataset, batch_size=model_config['batch_size'], shuffle=False, collate_fn=my_collator, num_workers=3)
            trainer.use_audio=use_audio
            trainer.test(model=model, dataloaders=test_loader)

            #kill logger
            logger.removeHandler(file_handler)
            file_handler.close()
    

def debug():
    # INSERT_YOUR_CODE
    from utils import get_model_config
    import sys

    model = "AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p0.5_ft_encoder-lr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m"
    model = "AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0-aliattention-h4_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m"
    model = ""
    # model = "A_wavlm-base-plus-cnn-TinyLlama-bs1_ft_encoder_mp10_str2_lr0.0001_T_avdeepfake1m"
    # model = "A_wavlm-base-plus-cnn-TinyLlama-bs1_lora16-16_ft_encoder-lr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc"
    # model = "T_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p0.5_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m"
    # # 保存原始sys.argv
    _orig_argv = sys.argv
    # attention version
    # sys.argv = [
    #     sys.argv[0],
    #     "--encoder", "microsoft/wavlm-base-plus",
    #     "--connector", "cnn",
    #     "--llm", "TinyLlama-1.1B-Chat-v1.0",
    #     "--batch-size", "1",
    #     "--meanpool", "100",
    #     "--use-config", "avdeepfake1m_llamapartialspoof_fakeloc_debug.json",
    #     "--checkpoints", f"checkpoints/{model}/last.ckpt",
    #     "--epoch-to-test", "0",
    #     '--test-on', 'AT',
    #     '--audio-transcript-align', 'attention',
    #     '--num-heads', '4',
    # ]

    model = "checkpoints/AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0-aliattention-h4_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m"
    config_file = "avdeepfake1m_avdeepfake1m_fakeloc_debug.json"
    test_on = "AT"

    _orig_argv = sys.argv
    sys.argv = [
        sys.argv[0],
        "--encoder", "microsoft/wavlm-base-plus",
        "--connector", "cnn",
        "--llm", "TinyLlama-1.1B-Chat-v1.0",
        "--batch-size", "1",
        "--meanpool", "100",
        "--use-config", config_file,
        "--checkpoints", f"checkpoints/{model}/last.ckpt",
        "--epoch-to-test", "0",
        "--test-on", test_on,
        ]

    model = "A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc_new"
    model = "A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc"
    # model = "AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m"
    config_file = "avdeepfake1m_avdeepfake1m_fakeloc_debug.json"
    test_on = "A"

    _orig_argv = sys.argv
    sys.argv = [
        sys.argv[0],
        "--encoder", "microsoft/wavlm-base-plus",
        "--connector", "cnn",
        "--llm", "TinyLlama-1.1B-Chat-v1.0",
        "--batch-size", "1",
        "--meanpool", "100",
        "--use-config", config_file,
        "--checkpoints", f"checkpoints/{model}/last.ckpt",
        "--epoch-to-test", "0",
        "--test-on", test_on,
    ]

    model = "A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m_fakeloc"
    sys.argv = [
        sys.argv[0],
        "--encoder", "microsoft/wavlm-base-plus",
        "--connector", "cnn",
        "--llm", "TinyLlama-1.1B-Chat-v1.0",
        "--batch-size", "1",
        "--meanpool", "100",
        "--use-config", "avdeepfake1m_avdeepfake1m_fakeloc_debug.json",
        "--checkpoints", f"checkpoints/{model}/last.ckpt",
        "--epoch-to-test", "0",
        '--test-on', 'A',
   ]


    # # fake token prediction version
    # model = "A_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_tokens-[fake]_avdeepfake1m_space_faketoken"
    # sys.argv = [
    #     sys.argv[0],
    #     "--encoder", "microsoft/wavlm-base-plus",
    #     "--connector", "cnn",
    #     "--llm", "TinyLlama-1.1B-Chat-v1.0",
    #     "--batch-size", "1",
    #     "--meanpool", "100",
    #     "--use-config", "avdeepfake1m_avdeepfake1m_fakeloc_debug.json",
    #     "--checkpoints", f"checkpoints/{model}/last.ckpt",
    #     "--epoch-to-test", "0",
    #     '--test-on', 'A',
    #     '--add-special-tokens', "[fake]",
    #     '--extract-fake-probs'
    # ]

    # # #AT
    # model = "AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0-aliconcat-h0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_tokens-[fake]_avdeepfake1m_space_faketoken"
    # sys.argv = [
    #     sys.argv[0],
    #     "--encoder", "microsoft/wavlm-base-plus",
    #     "--connector", "cnn",
    #     "--llm", "TinyLlama-1.1B-Chat-v1.0",
    #     "--batch-size", "1",
    #     "--meanpool", "100",
    #     "--use-config", "avdeepfake1m_avdeepfake1m_fakeloc_debug.json",
    #     "--checkpoints", f"checkpoints/{model}/last.ckpt",
    #     "--epoch-to-test", "0",
    #     '--test-on', 'AT',
    #     '--add-special-tokens', "[fake]",
    #     '--extract-fake-probs',
    #     "--test-use-batch-sampling", "--test-do-sample", "--test-top-p", "0.95", "--test-temperature", "2", "--test-top-k", "20"
    # ]
    # "--use-config", "avdeepfake1m_llamapartialspoof_fakeloc_debug.json",



    # full fine-tuning with the same config for debugging
    model="AT_wavlm-base-plus-cnn-TinyLlama-bs1_fullftllm_p1.0-aliconcat-h0_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_tokens-[fake]_ftw100.0_llamapartialspoof_space_faketoken"
    # model="A_wavlm-base-plus-cnn-TinyLlama-bs1_fullftllm_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_tokens-[fake]_ftw100.0_llamapartialspoof_space_faketoken"
    # For test debugging, move the output .txt if it exists to avoid overwriting.

    config_file = "avdeepfake1m_avdeepfake1m_fakeloc_debug.json"
    args = [
        'test.py', # The first argument is always the script name
        '--encoder', 'microsoft/wavlm-base-plus',
        '--connector', 'cnn',
        '--llm', 'TinyLlama-1.1B-Chat-v1.0',
        '--batch-size', '1',
        '--meanpool', '100',
        '--use-config', 'avdeepfake1m_llamapartialspoof_fakeloc_debug.json',
        '--checkpoint', f"checkpoints/{model}/last.ckpt" ,
        '--epoch-to-test', '0',
        '--test-on', 'AT',
        '--full-finetune-llm',
        '--add-special-tokens', '[fake]',
        '--extract-fake-probs'
    ]


    # 用 sys.argv 更新 model_config
    model_config = get_model_config()

    sys.argv = _orig_argv  # 恢复 sys.argv

    print("Debug parameters from sys.argv:")
    for k, v in model_config.items():
        print(f"{k}: {v}")

    main(model_config)

def debug_sampling_inference():
    """Print expected log and batch-sample paths without running inference."""
    import sys

    model = "checkpoints/AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p1.0-aliattention-h4_ft_encoder-enlr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m"
    config_file = "avdeepfake1m_avdeepfake1m_fakeloc_debug.json"
    test_on = "AT"

    _orig_argv = sys.argv
    sys.argv = [
        sys.argv[0],
        "--encoder", "microsoft/wavlm-base-plus",
        "--connector", "cnn",
        "--llm", "TinyLlama-1.1B-Chat-v1.0",
        "--batch-size", "1",
        "--meanpool", "100",
        "--use-config", config_file,
        "--checkpoints", f"checkpoints/{model}/last.ckpt",
        "--epoch-to-test", "0",
        "--test-on", test_on,
        "--test-use-batch-sampling", "--test-do-sample", "--test-top-p", "0.9", "--test-temperature", "0.8", "--test-top-k", "20"
    ]
    model_config = get_model_config()
    sys.argv = _orig_argv

    if model_config['test_on'] == 'AT':
        # text_options = [("AT", True, True), ("T", True, False), ("A", False, True)]
        text_options = [("AT", True, True)]
    elif model_config['test_on'] == 'T':
        text_options = [('T', True, False)]
    else:
        text_options = [('A', False, True)]
    print("Debug parameters from sys.argv:")
    for k, v in model_config.items():
        print(f"{k}: {v}")

    main(model_config)



if __name__ == "__main__":
    model_config = get_model_config()
    main(model_config)
    # debug()
    # debug_sampling_inference()

