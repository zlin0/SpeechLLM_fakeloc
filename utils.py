import argparse
import json
import time


LLM_SPECS = {
    "TinyLlama-1.1B-Chat-v1.0": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2048),
    "Llama-3.2-1B-Instruct": ("meta-llama/Llama-3.2-1B-Instruct", 2048),
    "Qwen2.5-1.5B-Instruct": ("Qwen/Qwen2.5-1.5B-Instruct", 1536),
    "Phi-3-mini-4k-instruct": ("microsoft/Phi-3-mini-4k-instruct", 3072),
    "gemma-2-2b-it": ("google/gemma-2-2b-it", 2304),
}


def resolve_llm_name_and_dim(llm_arg, llm_dim_arg):
    if llm_arg in LLM_SPECS:
        llm_name, default_dim = LLM_SPECS[llm_arg]
    elif '/' in llm_arg:
        llm_name, default_dim = llm_arg, None
    else:
        supported = ', '.join(sorted(LLM_SPECS.keys()))
        raise ValueError(
            f"Unknown --llm '{llm_arg}'. Use one of [{supported}] or pass a full HuggingFace repo id (owner/model)."
        )

    if int(llm_dim_arg) > 0:
        llm_dim = int(llm_dim_arg)
    elif default_dim is not None:
        llm_dim = int(default_dim)
    else:
        raise ValueError(
            f"Cannot infer hidden size for --llm '{llm_arg}'. Please provide --llm-dim explicitly."
        )

    return llm_name, llm_dim

def get_config(args):
    if args.use_config is None: #use default config
        datasets = {
        "train":['librispeech_train-clean-100', 'iemocap_ses01-03', 'CV-EN_train', 'MSP_Podcast_Train', 'voxceleb2_enriched_dev'],
        "dev":['librispeech_dev-clean', 'iemocap_ses04', 'CV-EN_dev', 'MSP_Podcast_Validation', 'voxceleb2_enriched_test'],
        "test":['librispeech_test-clean', 'iemocap_ses05', 'CV-EN_test', 'MSP_Podcast_Test', 'voxceleb2_enriched_test'],
        }
        datasets = {split:{data:[] for data in datasets[split]} for split in datasets} #empty list means use all available fields for that dataset
    else:
        with open(f'config/{args.use_config}', 'r') as file:
            datasets = json.load(file)

    use_summaries = False
    for data in datasets['dev']:
        if "summary" in datasets['dev'][data]:
            use_summaries=True
            break
    return datasets, use_summaries



def get_model_config():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='microsoft/wavlm-base-plus')  
    parser.add_argument('--connector', default='cnn' )  
    parser.add_argument('--llm', default='TinyLlama-1.1B-Chat-v1.0')  
    parser.add_argument('--llm-dim', default=-1, type=int,
                        help="LLM hidden size. If <=0, use built-in value for known aliases.")
    parser.add_argument('--connector-k', default=2, type=int)
    parser.add_argument('--connector-dim', default=512, type=int)
    parser.add_argument('--encoder-dim', default=768, type=int)
    parser.add_argument('--connector-layers', default=1, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--truncate-sec', default=-1, type=int)
    parser.add_argument('--lr', default=1.0, type=float)
    parser.add_argument('--encoder-lr', default=-1, type=float)
    parser.add_argument("--no-lora", action='store_true')
    parser.add_argument("--full-finetune-llm", action='store_true',
                        help="Fully fine-tune all LLM parameters (disables LoRA)")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--ft-encoder", action='store_true')
    parser.add_argument("--use-text", action='store_true')
    parser.add_argument("--prob-text", default=0.5, type=float)
    parser.add_argument("--no-audio", action='store_true')
    parser.add_argument('--epoch-to-test', default=1, type=int)
    parser.add_argument("--meanpool", default=1, type=int)
    parser.add_argument("--total-training-epoch", default=1000, type=int)
    parser.add_argument("--use-config", default='avdeepfake1m_llamapartialspoof_fakeloc.json', type=str)
    parser.add_argument("--group", default='Localization', type=str)
    parser.add_argument("--nickname", default='_T_debug', type=str)
    parser.add_argument("--test-on", default='A', type=str)
    parser.add_argument("--max_size_per_dev_set", type=int, default=50)
    parser.add_argument("--num-heads", type=int, default=0)
    parser.add_argument("--audio-transcript-align", type=str, default=None)
    parser.add_argument("--add-special-tokens", nargs='+', default=None, help="Special tokens to add (e.g., --add-special-tokens '[fake]' '[real]')")
    parser.add_argument("--fake-token-loss-weight", type=float, default=1.0,
                        help="Increase loss weight for [fake] token (1.0 = no change)")
    parser.add_argument("--trainer-variant", type=str, default="baseline", choices=["baseline", "multitask"],
                        help="Select trainer implementation")
    parser.add_argument("--realfake-loss-weight", type=float, default=0.5,
                        help="Auxiliary loss weight for frame-level Real/Fake classification (multitask only)")
    parser.add_argument("--frame-pred-stage", type=str, default="pre_connector", choices=["post_connector", "pre_connector"],
                        help="For multitask frame-level prediction, use embeddings before or after connector")
    parser.add_argument("--max-audio-duration", type=float, default=500, help="Maximum audio duration in seconds. Samples longer than this will be filtered during training (default: -1 = no limit)")
    #parser.add_argument("--checkpoints", type=str, default="checkpoints/AT_wavlm-base-plus-cnn-TinyLlama-bs1_lora8-16_p0.5_ft_encoder-lr2e-06_mp10_str2_lr0.0001_T_avdeepfake1m/last.ckpt")
    parser.add_argument("--checkpoints", type=str, default="")
 
    # Testing parameters for generation --------- 
    parser.add_argument("--extract-fake-probs", action='store_true', help="Extract [fake] token probabilities during testing")

    parser.add_argument("--test-use-batch-sampling", action='store_true', help="Use batch sampling during test_step instead of single generation")
    parser.add_argument("--test-num-samples", type=int, default=1, help="Number of samples to generate when using batch sampling")
    parser.add_argument("--test-max-new-tokens", type=int, default=2048, help="Max new tokens for test generation")
    parser.add_argument("--test-do-sample", action='store_true', help="Enable sampling for test generation")
    parser.add_argument("--test-top-p", type=float, default=1.0, help="Top-p for test sampling (None to disable)")
    parser.add_argument("--test-top-k", type=int, default=50, help="Top-k for test sampling")
    parser.add_argument("--test-temperature", type=float, default=1.0, help="Temperature for test sampling")
    
    # Few-shot in-context learning parameters for testing
    parser.add_argument("--icl-num-shots", type=int, default=0, help="Number of few-shot examples to prepend from a train CSV during testing")
    parser.add_argument("--icl-train-csv", type=str, default=None, help="CSV file used to sample few-shot in-context examples")
    parser.add_argument("--icl-seed", type=int, default=42, help="Random seed for selecting in-context examples")

    args = parser.parse_args()

    # Training Parameters
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    use_lora = (not args.no_lora) and (not args.full_finetune_llm)

    # Model naming
    llm_name, llm_dim = resolve_llm_name_and_dim(args.llm, args.llm_dim)
    llm_tag = llm_name.split('/')[-1].split('-')[0]

    model_name = f"{args.encoder.split('/')[-1]}-{args.connector}-{llm_tag}-bs{batch_size}"
    if args.full_finetune_llm:
        model_name = model_name+'_fullftllm'
    elif args.no_lora:
        model_name = model_name+'_nolora'
    else:
        model_name = model_name+f"_lora{args.lora_r}-{args.lora_alpha}"

    if args.use_text:
        model_name = model_name + f"_p{float(args.prob_text)}"
        if args.no_audio: model_name = 'T_'+model_name
        else: model_name = 'AT_'+model_name
        if args.audio_transcript_align:
            model_name = model_name + f"-ali{args.audio_transcript_align}" + f"-h{args.num_heads}"
    else:
        if args.no_audio: exit("not using text nor audio!")
        else: model_name = 'A_'+model_name

    if args.encoder_lr==-1: args.encoder_lr = float(args.lr)/50
    if args.ft_encoder: 
        model_name = model_name+'_ft_encoder'
        model_name = model_name+ f"-enlr{args.encoder_lr}"
    
    if args.meanpool!=1: model_name = model_name+f'_mp{args.meanpool}'
    if lr == 1.0: lr = 1e-4 if 'linear' not in args.connector else 1e-5
    if args.connector=='cnn':
        stride = int(args.connector_k)*(int(args.connector_k)//2)*(int(args.connector_k)//2)
        model_name = model_name+f'_str{stride}'
    model_name =  f"{model_name}_lr{lr}"
    # if args.extract_fake_probs: # only related to inference, not training, so not adding to model name for now
    #     model_name = f"{model_name}_[fake]"
    if args.add_special_tokens is not None:
        tokens_str = '_'.join([token.replace(' ', '_') for token in args.add_special_tokens])
        model_name = f"{model_name}_tokens-{tokens_str}"
        if args.fake_token_loss_weight != 1.0:
            model_name = f"{model_name}_ftw{args.fake_token_loss_weight}"
    if args.trainer_variant == "multitask":
        model_name = f"{model_name}_mt-{args.frame_pred_stage.split('_')[0]}-rfw{args.realfake_loss_weight}-{args.frame_pred_stage}"


    if args.nickname!='_': model_name = model_name + str(args.nickname)
    # Wandb params
    log_path = f"logs/{model_name}"
    log_path = f"logs/{model_name}_{time.strftime('%Y%m%d_%H%M')}"
    
    group = args.group

    # Encoder
    audio_encoder_name=args.encoder
    if args.encoder=='MFCC':
        audio_enc_dim = 80
    else:
        audio_enc_dim = int(args.encoder_dim)
    
    # LLM
    
    # Special tokens are already a list from nargs='+'
    add_special_tokens = args.add_special_tokens

    # Datasets
    datasets, use_summaries = get_config(args)

    # Get all infos
    model_config = {
                'audio_enc_dim':audio_enc_dim, 
                'llm_dim': llm_dim, 
                'audio_encoder_name': audio_encoder_name, 
                'connector_name': args.connector,
                'llm_name': llm_name,
                'finetune_encoder': args.ft_encoder,
                'connector_k': int(args.connector_k),
                'connector_dim': int(args.connector_dim),
                'connector_layers': int(args.connector_layers),
                'meanpool':int(args.meanpool),
                'use_lora': use_lora,
                'full_finetune_llm': args.full_finetune_llm,
                'use_text':args.use_text,
                'prob_text':float(args.prob_text),
                'use_audio':not args.no_audio,
                'lora_r': args.lora_r,
                'lora_alpha': args.lora_alpha,
                'max_lr': lr,
                'enc_lr': args.encoder_lr,
                'batch_size':batch_size,
                'total_training_epoch': int(args.total_training_epoch),
                'warmup_steps': 100,
                'grad_accumulate_steps': 64//batch_size,
                'max_number_seconds': args.truncate_sec,
                'max_audio_duration': args.max_audio_duration,
                'train_batch_per_epoch': 4096,
                'train_sets':datasets['train'],
                'dev_sets':datasets['dev'],
                'test_sets':datasets['test'],
                'max_size_per_dev_set':args.max_size_per_dev_set,
                'log_path':log_path,
                'group':group,
                'model_name':model_name,
                'checkpoints':args.checkpoints,
                'epoch_to_test':int(args.epoch_to_test),
                'use_summaries':use_summaries,
                'test_on':str(args.test_on),
                'audio_transcript_align':str(args.audio_transcript_align),
                'num_heads':int(args.num_heads),
                'trainer_variant': args.trainer_variant,
                'realfake_loss_weight': float(args.realfake_loss_weight),
                'frame_pred_stage': args.frame_pred_stage,
                'add_special_tokens': add_special_tokens,
                'fake_token_loss_weight': float(args.fake_token_loss_weight),
                'extract_fake_probs': args.extract_fake_probs,
                'test_use_batch_sampling': args.test_use_batch_sampling,
                'test_num_samples': int(args.test_num_samples),
                'test_max_new_tokens': int(args.test_max_new_tokens),
                'test_do_sample': args.test_do_sample,
                'test_top_p': args.test_top_p,
                'test_top_k': int(args.test_top_k),
                'test_temperature': float(args.test_temperature),

                'icl_num_shots': int(args.icl_num_shots),
                'icl_train_csv': args.icl_train_csv,
                'icl_seed': int(args.icl_seed),
        }
    return model_config

