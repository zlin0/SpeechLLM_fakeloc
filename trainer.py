import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import wandb
import pytorch_lightning as pl
import numpy as np
from jiwer import wer
import torchmetrics
import random
import re
import json
import torch.nn.functional as F

from model.encoder import get_audio_encoder, TransformerAudioEncoder
from model.connector import get_connector, LinearConnector, LinearPoolConnector, CNNConnector
from model.llm import get_llm
from metrics import MAE
from rouge_score import rouge_scorer
# from evaluate import load
import logging
import pandas as pd
import ast
import os

from metrics import compute_wfler, convert_fake_transcript
from metrics_writer import write_wer_wfler_summary
from extract_fake_prob import extract_fake_probs_from_generation, extract_word_fake_probs_from_generation


def calculate_iou(target_region, predicted_region):
    """
    Calculate the Intersection over Union (IoU) between two sets of time regions.

    Parameters:
        target_region: List[List[float]] or str, the target regions, represented as a 2D array where each row is a [start, end] time segment.
        predicted_region: List[List[float]] or str, the predicted regions, represented as a 2D array where each row is a [start, end] time segment.

    Returns:
        IoU: float, the Intersection over Union value (ranging from 0 to 1).
    """
    # Initialize the intersection and union areas
    intersection = 0
    union = 0
    
    try:
        # Ensure target_region is a list
        if isinstance(target_region, str):
            target_region = ast.literal_eval(target_region)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Error parsing target_region: {e}")

    # Ensure predicted_region is a list, being robust to model output
    if isinstance(predicted_region, str):
        if predicted_region.lower() == 'na':
            predicted_region = []
        else:
            try:
                predicted_region = ast.literal_eval(predicted_region)
            except (ValueError, SyntaxError):
                predicted_region = []  # Malformed prediction string
    
    if not isinstance(target_region, list):
        target_region = []
    if not isinstance(predicted_region, list):
        predicted_region = []
    
    # Calculate intersection area
    for t_start, t_end in target_region:
        for p_start, p_end in predicted_region:
            # Compute the intersection range
            inter_start = max(t_start, p_start)
            inter_end = min(t_end, p_end)
            inter_duration = max(0, inter_end - inter_start)  # Intersection duration is 0 if there is no overlap
            
            intersection += inter_duration

    # Calculate the total area of the target regions
    target_area = sum(end - start for start, end in target_region)
    
    # Calculate the total area of the predicted regions
    predicted_area = sum(end - start for start, end in predicted_region)
    
    # Calculate the union area
    union = target_area + predicted_area - intersection
    
    # If the union is 0 (e.g., both regions are empty), IoU is defined as 0
    if union == 0:
        return 0.0
    
    # Calculate IoU
    iou = intersection / union
    return iou

class MeanPooler(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        seq_len = x.size(1)
        k = min(self.k, seq_len) if seq_len > 0 else 1
        return F.avg_pool1d(x.transpose(1, 2), kernel_size=k, stride=k).transpose(1, 2)


    # class AudioTextAttentionAligner(nn.Module):
    #     def __init__(self, embed_dim, attn_dim=None, num_heads=4):
    #         super().__init__()
    #         attn_dim = attn_dim or embed_dim
    #         self.num_heads = num_heads
    #         self.attn_dim = attn_dim
    #         assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
    #         self.head_dim = attn_dim // num_heads

    #         # Q, K, V projections (project to attn_dim)
    #         self.query_proj = nn.Linear(embed_dim, attn_dim)
    #         self.key_proj = nn.Linear(embed_dim, attn_dim)
    #         self.value_proj = nn.Linear(embed_dim, attn_dim)

    #         self.out_proj = nn.Linear(attn_dim, attn_dim)

    #     def forward(self, speech_embeds, transcript_embeds):
    #         # speech_embeds: (batch, seq_audio, hidden_dim)
    #         # transcript_embeds: (batch, seq_text, hidden_dim)

    #         B, S_audio, _ = speech_embeds.shape
    #         B, S_text, _ = transcript_embeds.shape

    #         Q = self.query_proj(speech_embeds)      # (batch, seq_audio, attn_dim)
    #         K = self.key_proj(transcript_embeds)    # (batch, seq_text, attn_dim)
    #         V = self.value_proj(transcript_embeds)  # (batch, seq_text, attn_dim)

    #         # Reshape to (batch, n_head, seq_len, head_dim)
    #         def split_heads(x, seq_len):
    #             return x.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    #             # (batch, num_heads, seq_len, head_dim)

    #         Q = split_heads(Q, S_audio)
    #         K = split_heads(K, S_text)
    #         V = split_heads(V, S_text)

    #         # Attention scores: (batch, num_heads, seq_audio, seq_text)
    #         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
    #         attn_weights = F.softmax(attn_scores, dim=-1)

    #         # Output: (batch, num_heads, seq_audio, head_dim)
    #         attended = torch.matmul(attn_weights, V)

    #         # Concatenate heads back: (batch, seq_audio, attn_dim)
    #         attended = attended.transpose(1, 2).contiguous().view(B, S_audio, self.attn_dim)
    #         attended_audio = self.out_proj(attended)

    #         # Optionally, return average attention over heads
    #         avg_attn_weights = attn_weights.mean(dim=1)  # (batch, seq_audio, seq_text)

    #         return attended_audio, avg_attn_weights

    
    # Example function to use the aligner:
    # (You can create the aligner module globally in your class and call it)
    def align_audio_transcript(speech_embeds, transcript_embeds, aligner):
        return aligner(speech_embeds, speech_embeds, transcript_embeds)

def sanitize_field(s):
    if s is None:
        return ""
    return s.replace("\n", " ").replace("\r", " ")


class SpeechLLMLightning(pl.LightningModule):
    def __init__(self, 
                 audio_enc_dim=512, 
                 llm_dim=2048, 
                 audio_encoder_name="speech-tokenizer",
                 connector_name='linear-pool',
                 llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 finetune_encoder=False,
                 use_audio=True,
                 connector_k=5,
                 connector_dim=512,
                 connector_layers=1,
                 meanpool=1,
                 use_lora=True,
                 full_finetune_llm=False,
                 lora_r=32,
                 lora_alpha=2,
                 max_lr=3e-4,
                 enc_lr=2e-6,
                 total_training_step=500000,
                 warmup_steps=1000,
                 audio_transcript_align=None,
                 num_heads=4,
                 add_special_tokens=None,
                 fake_token_loss_weight=1.0,
                 extract_fake_probs=False,
                 test_use_batch_sampling=False,
                 test_num_samples=1,
                 test_max_new_tokens=250,
                 test_do_sample=False,
                 test_top_p=None,
                 test_top_k=2,
                 test_temperature=0.8,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.audio_enc_dim = audio_enc_dim
        self.llm_dim = llm_dim
        self.llm_name = llm_name
        self.finetune_encoder = finetune_encoder and use_audio
        self.use_lora = use_lora
        self.full_finetune_llm = full_finetune_llm

        self.audio_transcript_align = audio_transcript_align
        self.aligner = None
        if (self.audio_transcript_align == "attention"):
            self.aligner = nn.MultiheadAttention(llm_dim, num_heads=num_heads, batch_first=True)

        print("Getting audioencoder.")
        self.audio_encoder = get_audio_encoder(audio_encoder_name, finetune_encoder)
        print("Getting connector.")
        self.connector = get_connector(connector_name, audio_enc_dim, llm_dim, connector_k, connector_dim, connector_layers)
        print("Getting pooling.")
        self.pooling = MeanPooler(k=meanpool)
        print("Getting llm tokenizer.")
        self.llm_tokenizer, self.llm_model = get_llm(
            llm_name,
            use_lora,
            lora_r,
            lora_alpha,
            add_special_tokens,
            full_finetune_llm,
        )
        print("Finish initialize speechLLM")

        self.fake_token = "[fake]"
        self.fake_token_loss_weight = float(fake_token_loss_weight)
        self.fake_token_id = None
        try:
            vocab = self.llm_tokenizer.get_vocab()
            if self.fake_token in vocab:
                self.fake_token_id = self.llm_tokenizer.convert_tokens_to_ids(self.fake_token)
            else:
                logging.warning("[fake] token not found in tokenizer vocab; fake_token_loss_weight will be ignored.")
        except Exception as exc:
            logging.warning(f"Unable to resolve [fake] token id: {exc}. fake_token_loss_weight will be ignored.")
        
        self.extract_fake_probs = extract_fake_probs
        self.icl_num_shots = int(kwargs.get("icl_num_shots", 0) or 0)
        self.icl_seed = int(kwargs.get("icl_seed", 42) or 42)
        self.icl_train_csv = kwargs.get("icl_train_csv", None)
        # parameters for multiple sampling during testing 
        self.test_use_batch_sampling = test_use_batch_sampling
        self.test_num_samples = test_num_samples
        self.test_max_new_tokens = test_max_new_tokens
        self.test_do_sample = test_do_sample
        self.test_top_p = test_top_p
        self.test_top_k = test_top_k
        self.test_temperature = test_temperature
        
        self.max_lr = max_lr
        self.enc_lr = enc_lr
        self.total_training_step = total_training_step
        self.warmup_steps = warmup_steps
        self.use_embedding_loss = False
        self.num_validation_samples = 5000
        self.use_audio = use_audio

        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL', 'rouge2'], use_stemmer=True)
        # self.bert_scorer = load("bertscore")
        
    def configure_optimizers(self):
        opt = [
            {"params": self.audio_encoder.parameters(), "lr": self.enc_lr if (self.finetune_encoder and self.use_audio) else 0},
            {"params": self.connector.parameters(), "lr": self.max_lr if self.use_audio else 0},
            {"params": self.llm_model.parameters(), "lr": self.max_lr if (self.use_lora or self.full_finetune_llm) else 0},
        ]
        optimizer = Adam(opt, lr=self.max_lr)
        return optimizer
        

    def encode_speech_segment(self, mel):
        if self.finetune_encoder:
            speech_embeds = self.audio_encoder(mel)
        else:
            with torch.no_grad():
                speech_embeds = self.audio_encoder(mel)
        # print(f"encoded size: {speech_embeds.shape}")
        return speech_embeds

    def encode(self, mel, pre_tokenized_ids, transcript_tokenized_ids, 
               post_tokenized_ids, output_tokenized_ids, 
               return_embedding_loss=False, chunk_size=60*16_000, test_mode=False,
               return_speech_embeds=False):
        batch_size = mel.shape[0]
        
        speech_embeds_raw = None
        speech_embeds = None
        if self.use_audio:
            # WavLM's first convolution has a kernel size of 10 and stride of 5.
            # A minimum length is required to avoid errors. 400 is a safe value.
            if mel.shape[1] > 400:
                if mel.shape[1] < chunk_size:
                    speech_embeds_raw = self.encode_speech_segment(mel)
                else:
                    chunks = mel.split(chunk_size, dim=1)
                    outs = []
                    for c in chunks:
                        if c.shape[1] >= 16_000:  # WavLM needs at least one second of audio
                            outs.append(self.encode_speech_segment(c))
                    
                    if outs:
                        speech_embeds_raw = torch.cat(outs, dim=1)

                    del chunks, outs

            if speech_embeds_raw is not None:
                speech_embeds = self.connector(self.pooling(speech_embeds_raw))
        
        # If speech_embeds is still None, it means no valid audio was found/processed.

        if self.use_lora: embedder = self.llm_model.model.model.embed_tokens
        else: embedder = self.llm_model.model.embed_tokens

        pre_prompt_embeds = embedder(pre_tokenized_ids)
        if transcript_tokenized_ids.shape[1] > 0:
            transcript_embeds = embedder(transcript_tokenized_ids)
        else:
            transcript_embeds = None
        post_prompt_embeds = embedder(post_tokenized_ids)
        output_prompt_embeds = embedder(output_tokenized_ids)

        cat_embs = [pre_prompt_embeds]
        input_token_length = pre_tokenized_ids.shape[1]


        if self.use_audio and self.aligner and transcript_embeds is not None:
            if speech_embeds is not None: # Check if speech_embeds was created
                if(self.audio_transcript_align == "attention"):
                    speech_text_embeds, _ = self.aligner(speech_embeds, transcript_embeds, transcript_embeds) #[B, 20, 2048]
                    cat_embs.append(speech_text_embeds)
                    input_token_length+=speech_text_embeds.shape[1]
                else:
                    raise NotImplementedError
            # If speech_embeds is None, do nothing, just skip audio part

        elif self.use_audio and speech_embeds is not None: # Check if speech_embeds was created
            cat_embs.append(speech_embeds)
            input_token_length+=speech_embeds.shape[1]

        cat_embs.append(post_prompt_embeds)
        input_token_length+=post_prompt_embeds.shape[1]

        if not test_mode: cat_embs.append(output_prompt_embeds)

        combined_embeds = torch.cat(cat_embs, dim=1)
        atts = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(combined_embeds.device)

        label_ids = torch.cat([
            torch.ones([batch_size, input_token_length], device=combined_embeds.device)*-100,
            output_tokenized_ids
        ], 1).to(combined_embeds.device).to(torch.int64)

        if return_speech_embeds:
            return combined_embeds, atts, label_ids, speech_embeds_raw, speech_embeds

        return combined_embeds, atts, label_ids

    def forward(self, embeds, atts, label_ids):
        out = self.llm_model(
            inputs_embeds=embeds,
            attention_mask=atts,
            labels=label_ids,
        )
        return out

    def _compute_weighted_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        weights = torch.ones_like(shift_labels, dtype=per_token_loss.dtype)
        if self.fake_token_id is not None and self.fake_token_loss_weight != 1.0:
            weights = weights.masked_fill(shift_labels == self.fake_token_id, self.fake_token_loss_weight)
        weights = weights.masked_fill(shift_labels == -100, 0.0)
        weights = weights.view(-1)
        denom = weights.sum().clamp_min(1.0)
        return (per_token_loss * weights).sum() / denom

    def compute_loss(self, outputs, labels):
        if self.fake_token_id is None or self.fake_token_loss_weight == 1.0:
            return outputs["loss"]
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        return self._compute_weighted_loss(logits, labels)

    def generate(self, embeds, max_new_tokens=2048):
        out = self.llm_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=max_new_tokens,
        )
        return out
    
    def training_step(self, batch, batch_idx):
        audio_ids, mel, pre_tokenized_ids, transcript_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
        embeds, atts, label_ids = self.encode(mel, pre_tokenized_ids, transcript_tokenized_ids, post_tokenized_ids, output_tokenized_ids, test_mode=False)
        outputs = self.forward(embeds, atts, label_ids)
        loss = self.compute_loss(outputs, label_ids)
        self.log("train/loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio_ids, mel, pre_tokenized_ids, transcript_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
        embeds, atts, label_ids = self.encode(mel, pre_tokenized_ids, transcript_tokenized_ids, post_tokenized_ids, output_tokenized_ids, test_mode=False)
        outputs = self.forward(embeds, atts, label_ids)
        loss = self.compute_loss(outputs, label_ids)
        self.log("val/loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        # logits = outputs.logits
        # predicted_ids = torch.argmax(logits, dim=-1).cpu()
        embeds, _, _ = self.encode(mel, pre_tokenized_ids, transcript_tokenized_ids, post_tokenized_ids, output_tokenized_ids, test_mode=True)
        predicted_ids = self.generate(embeds=embeds).cpu()

        generated_output_text = self.llm_tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
        target_text = self.llm_tokenizer.decode(output_tokenized_ids[0], skip_special_tokens=False)
        
        extracted_pred = self.extract_prediction_values(generated_output_text)
        extracted_target = self.extract_prediction_values(target_text)

        self.get_keys_and_log(extracted_pred, extracted_target, v='val')

        if batch_idx in self.selected_samples_for_logging:
            sample_idx = self.selected_samples_for_logging.index(batch_idx)
            # Use wandb.log to log prediction and truth texts
            wandb.log({
                f"val_sample_{sample_idx}_pred": wandb.Html(f"<pre>{str(extracted_pred)}</pre>"), 
                f"val_sample_{sample_idx}_target": wandb.Html(f"<pre>{str(target_text).replace('<s>', '').replace('</s>', '')}</pre>"),
                f"val_sample_{sample_idx}_gen": wandb.Html(f"<pre>{generated_output_text.replace('<s>', '').replace('</s>', '')}</pre>"),
            }, commit=False)

        return {"val_loss": loss}
    

    
    ### def test_step(self, batch, batch_idx):
    def test_step(self, batch, batch_idx, save_logits=False):
        audio_ids, mel, pre_tokenized_ids, transcript_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
        logging.info(f"[ID]\t{audio_ids}") #only one

        embeds, atts, label_ids = self.encode(mel, pre_tokenized_ids, transcript_tokenized_ids, post_tokenized_ids, output_tokenized_ids, test_mode=True)
        
        # Extract [fake] probabilities if enabled
        fake_probs_data = None
        word_fake_probs_data = None
        if self.extract_fake_probs:
            generated_output_text, word_probs, token_probs = extract_word_fake_probs_from_generation(
                self, embeds, atts, max_new_tokens=250, special_token='[fake]'
            )
            # Store for later CSV saving
            fake_probs_data = token_probs
            word_fake_probs_data = word_probs
            # Also get predicted_ids for compatibility
            predicted_ids = self.llm_tokenizer.encode(generated_output_text, return_tensors='pt').to(embeds.device)
        else:
            if self.test_use_batch_sampling:
                if self.test_num_samples > 1 and not self.test_do_sample:
                    logging.warning(
                        "test_num_samples > 1 requires do_sample=True for diversity; overriding test_do_sample to True."
                    )
                    self.test_do_sample = True
                    
                gen_cfg = {
                    "max_new_tokens": self.test_max_new_tokens,
                    "do_sample": self.test_do_sample,
                    "temperature": self.test_temperature,
                    "num_return_sequences": self.test_num_samples,
                }
                if self.test_top_p is not None:
                    gen_cfg["top_p"] = self.test_top_p
                if self.test_top_k is not None:
                    gen_cfg["top_k"] = self.test_top_k
                predicted_ids = self.llm_model.generate(
                    inputs_embeds=embeds,
                    attention_mask=atts,
                    **gen_cfg,
                ).cpu()
                generated_texts = [
                    self.llm_tokenizer.decode(predicted_ids[i], skip_special_tokens=False)
                    for i in range(predicted_ids.shape[0])
                ]

                # Save batch samples to CSV with params in filename
                top_p_tag = "none" if self.test_top_p is None else str(self.test_top_p)
                file_name = (
                    f"batch_samples_ns{self.test_num_samples}"
                    f"_k{self.test_top_k}_p{top_p_tag}_t{self.test_temperature}.csv"
                )
                base_dir = getattr(self, "test_pred_base_path", None)
                if base_dir is None:
                    base_dir = "."
                os.makedirs(base_dir, exist_ok=True)
                batch_samples_path = os.path.join(base_dir, file_name)

                rows = []
                for idx, text in enumerate(generated_texts):
                    rows.append(
                        {
                            "audio_id": audio_ids[0],
                            "sample_idx": idx,
                            "text": text,
                        }
                    )
                write_header = not os.path.exists(batch_samples_path)
                pd.DataFrame(rows).to_csv(
                    batch_samples_path,
                    mode="a",
                    index=False,
                    header=write_header,
                    encoding="utf-8",
                )
                generated_output_text = generated_texts[0] if generated_texts else ""
                # if len(generated_texts) > 1:
                #     logging.info(f"[BATCH SAMPLES]\t{generated_texts}")
            else:
                predicted_ids = self.generate(embeds=embeds).cpu()
                generated_output_text = self.llm_tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
        
        # logits = outputs.logits
        # predicted_ids = torch.argmax(logits, dim=-1)

        input_token_length = output_tokenized_ids.shape[1]
        target_text = self.llm_tokenizer.decode(output_tokenized_ids[0], skip_special_tokens=False)
        # generated_output_text already set above based on extract_fake_probs flag

        # -------- Strict vs relaxed parsing --------
        # Strict: 只用 JSON 解析，不做兜底
        strict_extracted_target = self.extract_dictionary(target_text)
        strict_extracted_pred = self.extract_dictionary(generated_output_text)

        # Relaxed: 允许 fallback（当前 extract_prediction_values 已包含宽松逻辑）
        extracted_target = self.extract_prediction_values(target_text)
        extracted_pred_relaxed = self.extract_prediction_values(generated_output_text)

        # -------- Relaxed transcript & metrics --------
        if 'Fake_transcript' in extracted_target:
            target_transcript_relaxed = extracted_target['Fake_transcript']
        elif 'Transcript' in extracted_target:
            target_transcript_relaxed = extracted_target['Transcript']
        else:
            target_transcript_relaxed = ''
        
        if 'Fake_transcript' in extracted_pred_relaxed:
            predicted_transcript_relaxed = extracted_pred_relaxed['Fake_transcript']
        elif 'Transcript' in extracted_pred_relaxed:
            predicted_transcript_relaxed = extracted_pred_relaxed['Transcript']
        else:
            predicted_transcript_relaxed = ''

        gt_fakeword_vec_relaxed, target_transcript_notag_relaxed = convert_fake_transcript(target_transcript_relaxed)
        hy_fakeword_vec_relaxed, predicted_transcript_notag_relaxed = convert_fake_transcript(predicted_transcript_relaxed)
        if len(gt_fakeword_vec_relaxed) > 0 and len(hy_fakeword_vec_relaxed) > 0:
            wer_relaxed, wfler_relaxed = compute_wfler(
                target_transcript_notag_relaxed,
                gt_fakeword_vec_relaxed,
                predicted_transcript_notag_relaxed,
                hy_fakeword_vec_relaxed,
            )
        else:
            wer_relaxed = 'miss'
            wfler_relaxed = 'miss'

        # -------- Strict transcript & metrics（仅当严格解析成功时） --------
        if 'Fake_transcript' in strict_extracted_target:
            target_transcript_strict = strict_extracted_target['Fake_transcript']
        elif 'Transcript' in strict_extracted_target:
            target_transcript_strict = strict_extracted_target['Transcript']
        else:
            target_transcript_strict = ''

        if 'Fake_transcript' in strict_extracted_pred:
            predicted_transcript_strict = strict_extracted_pred['Fake_transcript']
        elif 'Transcript' in strict_extracted_pred:
            predicted_transcript_strict = strict_extracted_pred['Transcript']
        else:
            predicted_transcript_strict = ''

        if target_transcript_strict and predicted_transcript_strict:
            gt_fakeword_vec_strict, target_transcript_notag_strict = convert_fake_transcript(target_transcript_strict)
            hy_fakeword_vec_strict, predicted_transcript_notag_strict = convert_fake_transcript(predicted_transcript_strict)
            if len(gt_fakeword_vec_strict) > 0 and len(hy_fakeword_vec_strict) > 0:
                wer_strict, wfler_strict = compute_wfler(
                    target_transcript_notag_strict,
                    gt_fakeword_vec_strict,
                    predicted_transcript_notag_strict,
                    hy_fakeword_vec_strict,
                )
            else:
                wer_strict = 'miss'
                wfler_strict = 'miss'
        else:
            predicted_transcript_strict = ''
            wer_strict = 'miss'
            wfler_strict = 'miss'

        # 保存预测到和logging一样的路径，改后缀为pred
        # 获取当前logging的文件路径
        
        log_file_path = None
        for handler in logging.getLogger().handlers:
            if hasattr(handler, "baseFilename"):
                log_file_path = handler.baseFilename
                break
        if log_file_path is not None:
            base_no_ext = log_file_path.rsplit('.', 1)[0]
            if self.icl_num_shots > 0:
                base_dir = os.path.dirname(base_no_ext)
                base_name = os.path.basename(base_no_ext)
                src_name = "none"
                if self.icl_train_csv:
                    src_name = os.path.splitext(os.path.basename(str(self.icl_train_csv)))[0]
                src_name = src_name.replace(" ", "_")
                icl_dir = os.path.join(base_dir, "icl")
                os.makedirs(icl_dir, exist_ok=True)
                pred_file_path = os.path.join(
                    icl_dir,
                    f"{base_name}_icl-n{self.icl_num_shots}_seed{self.icl_seed}_src-{src_name}.csv"
                )
            else:
                pred_file_path = base_no_ext + ".csv"
        else:
            pred_file_path = "predictions.pred"  # fallback

        # 记住用于汇总指标的基路径（只需设置一次）
        if not hasattr(self, "test_pred_base_path"):
            self.test_pred_base_path = pred_file_path
            if(self.test_use_batch_sampling):
                self.test_pred_base_path = self.test_pred_base_path + "batch_sample"

        # df.to_csv(f"{save_dir}/llamapartialspoof.csv", index=False, header=True)
        # 将当前batch的输出结果追加到csv文件

        # 构建一行结果，与json_record类似，但以csv形式
        row = {
            "audio_id": audio_ids[0],
            "target": target_text,
            "extracted_target": str(target_transcript_relaxed),
            "predicted_raw": generated_output_text,
            "predicted_relaxed": str(predicted_transcript_relaxed),
            "wer_relaxed": wer_relaxed,
            "wfler_relaxed": wfler_relaxed,
            "predicted_strict": str(predicted_transcript_strict),
            "wer_strict": wer_strict,
            "wfler_strict": wfler_strict
        }

        # 如果是第一个batch或文件不存在，则写header，否则追加
        write_header = (batch_idx == 0 or (not os.path.exists(pred_file_path)))
        mode = "w" if write_header else "a"
        df = pd.DataFrame([row])
        df.to_csv(pred_file_path, mode=mode, index=False, header=write_header, encoding="utf-8")
        
        # Save [fake] probabilities if extracted
        if self.extract_fake_probs and fake_probs_data is not None:
            fake_probs_file = pred_file_path.rsplit('.', 1)[0] + "_fake_probs.csv"
            fake_rows = []
            for token, prob in fake_probs_data:
                fake_rows.append({
                    "audio_id": audio_ids[0],
                    "token": token,
                    "fake_prob": prob
                })
            fake_df = pd.DataFrame(fake_rows)
            fake_write_header = (batch_idx == 0 or (not os.path.exists(fake_probs_file)))
            fake_mode = "w" if fake_write_header else "a"
            fake_df.to_csv(fake_probs_file, mode=fake_mode, index=False, header=fake_write_header, encoding="utf-8")

        # Save word-level [fake] probabilities if extracted
        if self.extract_fake_probs and word_fake_probs_data is not None:
            word_fake_probs_file = pred_file_path.rsplit('.', 1)[0] + "_word_fake_probs.csv"
            word_rows = []
            for word, prob in word_fake_probs_data:
                word_rows.append({
                    "audio_id": audio_ids[0],
                    "word": word,
                    "fake_prob": prob
                })
            word_df = pd.DataFrame(word_rows)
            word_write_header = (batch_idx == 0 or (not os.path.exists(word_fake_probs_file)))
            word_mode = "w" if word_write_header else "a"
            word_df.to_csv(word_fake_probs_file, mode=word_mode, index=False, header=word_write_header, encoding="utf-8")
        
        # 在内存中累积每个样本的指标，供 epoch 结束时整体写入 .wer_wfler

        # 这里同时存入self.test_wer_records以便之后写入wer_wfler
        # per utterance.
        if not hasattr(self, "test_wer_records"):
            self.test_wer_records = []
        self.test_wer_records.append(
            {
                "audio_id": audio_ids[0],
                "wer_relaxed": wer_relaxed,
                "wfler_relaxed": wfler_relaxed,
                "wer_strict": wer_strict,
                "wfler_strict": wfler_strict,
            }
        )

        # #Json
        # if not hasattr(self, "test_wer_records"):
        #     self.test_wer_records = []
        # self.test_wer_records.append(
        #     {
        #         "audio_id": audio_ids[0],
        #         "wer_relaxed": wer_relaxed,
        #         "wfler_relaxed": wfler_relaxed,
        #         "wer_strict": wer_strict,
        #         "wfler_strict": wfler_strict,
        #     }
        # )

        # # INSERT_YOUR_CODE
        # # Use json to record the result for easier downstream parsing.
 
        # json_pred_file_path = pred_file_path.rsplit('.', 1)[0] + ".json"

        # prediction_record = {
        #     "audio_id": audio_ids[0],
        #     "target": target_text,
        #     "extracted_target": str(target_transcript_relaxed),
        #     "predicted_raw": generated_output_text,
        #     "predicted_relaxed": str(predicted_transcript_relaxed),
        #     "wer_relaxed": wer_relaxed,
        #     "wfler_relaxed": wfler_relaxed,
        #     "predicted_strict": str(predicted_transcript_strict),
        #     "wer_strict": wer_strict,
        #     "wfler_strict": wfler_strict
        # }
        # # Write header when first batch or new file, else append
        # write_mode = "w" if batch_idx == 0 or (not os.path.exists(json_pred_file_path)) else "a"
        # with open(json_pred_file_path, write_mode, encoding="utf-8") as jf:
        #     jf.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")

        # # 仍然逐样本写 .pred 文件，方便对齐原始 text 与解析结果
        # import os
        # if batch_idx == 0 or (not os.path.exists(pred_file_path)):
        #     with open(pred_file_path, "w", encoding="utf-8") as f:
        #         # Header: 原始 target / predicted + 宽松解析 + 严格解析
        #         f.write(
        #             "audio_id|target|extracted_target|predicted_raw|"
        #             "predicted_relaxed|wer_relaxed|wfler_relaxed|"
        #             "predicted_strict|wer_strict|wfler_strict\n"
        #         )

        # with open(pred_file_path, "a", encoding="utf-8") as f:
        #     f.write(
        #         f"{audio_ids[0]}|"
        #         f"{sanitize_field(target_text)}|{sanitize_field(str(target_transcript_relaxed))}|"
        #         f"{sanitize_field(generated_output_text)}|"
        #         f"{sanitize_field(str(predicted_transcript_relaxed))}|{wer_relaxed}|{wfler_relaxed}|"
        #         f"{sanitize_field(str(predicted_transcript_strict))}|{wer_strict}|{wfler_strict}\n"
        #     )

        # 对日志与指标，仍使用宽松解析，避免严格解析失败导致全 miss
        self.get_keys_and_log(extracted_pred_relaxed, extracted_target, v='test')
        logging.info(f"[RAW OUTPUT]\t{generated_output_text}")
        logging.info(f"[PREDICTION]\t{extracted_pred_relaxed}")
        logging.info(f"[RAW TARGET]\t{target_text}")
        logging.info(f"[TARGET]\t{extracted_target}")

        return {"test_loss": 0}

    def on_test_epoch_end(self):
        """
        在整个 test 结束后，把 relax / strict 的整体结果
        保存到与 .pred 同路径、同前缀的 .pred.wer_wfler 文件中。
        """
        if not hasattr(self, "test_wer_records") or not self.test_wer_records:
            return

        base_path = getattr(self, "test_pred_base_path", None)
        write_wer_wfler_summary(base_path, self.test_wer_records)

        # 清理缓存，避免多次 test 混到一起
        del self.test_wer_records
    
    def get_keys_and_log(self, extracted_pred, extracted_target, v='val'):

        keys = extracted_target.keys()
        pred_keys = extracted_pred.keys()

        for key in keys:
            if key not in pred_keys:
                extracted_pred[key] = "NA"

        if 'RealFake' in keys:
            target_realfake = extracted_target['RealFake']
            predicted_realfake = extracted_pred['RealFake']
            self.log(f"{v}/realfake", float(target_realfake.lower()==predicted_realfake.lower()))

        if 'Fake_transcript' in keys:
            target_transcript = extracted_target['Fake_transcript']
            predicted_transcript = extracted_pred['Fake_transcript']
            # wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
            gt_fakeword_vec, target_transcript_notag = convert_fake_transcript(target_transcript)
            hy_fakeword_vec, predicted_transcript_notag = convert_fake_transcript(predicted_transcript)
            wer_metric, wfler_metric = compute_wfler(target_transcript_notag, gt_fakeword_vec, predicted_transcript_notag, hy_fakeword_vec)
            self.log(f"{v}/fakewordwer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"{v}/wfler", wfler_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if 'Fake_periods' in keys:
            target_transcript = extracted_target['Fake_periods']
            predicted_transcript = extracted_pred['Fake_periods']
            iou_metric = calculate_iou(target_transcript.lower(), predicted_transcript.lower())
            self.log(f"{v}/iou", iou_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if 'Transcript' in keys:
            target_transcript = extracted_target['Transcript']
            predicted_transcript = extracted_pred['Transcript']
            wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
            self.log(f"{v}/wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def on_validation_epoch_start(self, n=16):
        """Select n=16 random validation samples to log for each epoch."""
        # Ensure val_dataloaders is not empty and trainer is available
        if not self.trainer or not self.trainer.val_dataloaders:
            self.selected_samples_for_logging = []
            return
            
        num_batches = self.trainer.num_val_batches[0]
        if num_batches > 0:
            num_to_sample = min(n, num_batches)
            self.selected_samples_for_logging = random.sample(range(num_batches), num_to_sample)
        else:
            self.selected_samples_for_logging = []

    
    def extract_dictionary(self, input_string):
        # Find <s> tag position
        s_start = input_string.find('<s>')
        if s_start == -1:
            # No <s> tag, try to find JSON directly
            return self._extract_json_direct(input_string)
        
        # Find the first { after <s>
        brace_start = input_string.find('{', s_start)
        if brace_start == -1:
            return {}
        
        # Use brace counting to find the matching closing brace
        # This correctly handles nested braces and strings with quotes
        brace_count = 0
        brace_end = -1
        in_string = False
        escape_next = False
        
        for i in range(brace_start, len(input_string)):
            char = input_string[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = i
                        break
        
        if brace_end == -1:
            return {}
        
        # Extract the JSON string
        dict_string = input_string[brace_start:brace_end+1]
        
        # Clean up common JSON issues (trailing commas)
        dict_string = re.sub(r',(\s*[}\]])', r'\1', dict_string)
        
        try:
            return json.loads(dict_string)
        except json.JSONDecodeError as e:
            # Try to fix more issues
            # Remove trailing commas before closing braces/brackets
            dict_string = re.sub(r',\s*}', '}', dict_string)
            dict_string = re.sub(r',\s*]', ']', dict_string)
            try:
                return json.loads(dict_string)
            except json.JSONDecodeError:
                return {}
    
    def _extract_json_direct(self, input_string):
        """Extract JSON directly from string without <s> tags"""
        brace_start = input_string.find('{')
        if brace_start == -1:
            return {}
        
        # Use brace counting to find the matching closing brace
        brace_count = 0
        brace_end = -1
        in_string = False
        escape_next = False
        
        for i in range(brace_start, len(input_string)):
            char = input_string[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = i
                        break
        
        if brace_end == -1:
            return {}
        
        dict_string = input_string[brace_start:brace_end+1]
        dict_string = re.sub(r',(\s*[}\]])', r'\1', dict_string)
        
        try:
            return json.loads(dict_string)
        except json.JSONDecodeError:
            dict_string = re.sub(r',\s*}', '}', dict_string)
            dict_string = re.sub(r',\s*]', ']', dict_string)
            try:
                return json.loads(dict_string)
            except json.JSONDecodeError:
                return {}


    def extract_prediction_values(self, input_string):
        """
        Strict first: try JSON. If it fails (e.g., model未正确转义引号)，
        尝试宽松兜底，仅抠出 Fake_transcript，方便统计与日志。
        """
        parsed = self.extract_dictionary(input_string)
        if parsed:
            return parsed

        # Fallback: 针对常见不合法 JSON，手动提取 Fake_transcript
        fallback = self._extract_fake_transcript_fallback(input_string)
        return fallback if fallback else {}

    def _extract_fake_transcript_fallback(self, input_string):
        """
        兜底：从 <s> ... </s> 中用正则抓取 Fake_transcript 对应的字符串，
        不依赖 json.loads。仅在严格解析失败时使用。
        """
        # 先取出 <s>...</s> 中的内容
        m = re.search(r'<s>\s*(.*?)\s*</s>', input_string, re.DOTALL)
        content = m.group(1) if m else input_string

        # 尝试匹配 "Fake_transcript": "...."
        # 使用贪婪匹配直到最后一个双引号前，避免内部引号导致过早截断
        m = re.search(r'"Fake_transcript"\s*:\s*"(.*)"\s*,?\s*}?\s*$', content, re.DOTALL)
        if not m:
            # 再试一次更宽松：匹配到行尾或非贪婪
            m = re.search(r'"Fake_transcript"\s*:\s*"(.*?)"', content, re.DOTALL)
        if m:
            fake_text = m.group(1)
            return {"Fake_transcript": fake_text}
        return {}
 

    # def extract_prediction_values(self, input_string):
    #     """
    #     提取字符串内所有<s> ... </s>包裹的JSON字典，返回一个dict列表
    #     """
    #     pattern = r'<s>\s*(\{.*?\})\s*</s>'
    #     matches = re.findall(pattern, input_string, re.DOTALL)
    #     dicts = []
    #     for dict_string in matches:
    #         dict_string = re.sub(r',\s*}', '}', dict_string)
    #         try:
    #             dicts.append(json.loads(dict_string))
    #         except json.JSONDecodeError:
    #             dicts.append({})
    #     return dicts



    # def extract_dictionary(self, input_string):
    #     # Find the position of <s> tag
    #     s_start = input_string.find('<s>')
    #     if s_start == -1:
    #         # No <s> tag found, try to find JSON directly
    #         return self._extract_json_from_string(input_string)
        
    #     # Find the position after <s> tag
    #     content_start = s_start + 3
    #     # Find the first { after <s>
    #     brace_start = input_string.find('{', content_start)
    #     if brace_start == -1:
    #         return {}
        
    #     # Find the matching closing brace by counting braces
    #     brace_count = 0
    #     brace_end = -1
    #     for i in range(brace_start, len(input_string)):
    #         if input_string[i] == '{':
    #             brace_count += 1
    #         elif input_string[i] == '}':
    #             brace_count -= 1
    #             if brace_count == 0:
    #                 brace_end = i
    #                 break
        
    #     if brace_end == -1:
    #         return {}
        
    #     # Extract the JSON string
    #     dict_string = input_string[brace_start:brace_end+1]
        
    #     # Clean up common JSON issues
    #     dict_string = re.sub(r',\s*}', '}', dict_string)
    #     dict_string = re.sub(r',(\s*[}\]])', r'\1', dict_string)
        
    #     try:
    #         return json.loads(dict_string)
    #     except json.JSONDecodeError:
    #         return {}
    
    # def _extract_json_from_string(self, input_string):
    #     """Helper method to extract JSON from string without <s> tags"""
    #     brace_start = input_string.find('{')
    #     if brace_start == -1:
    #         return {}
        
    #     # Find the matching closing brace
    #     brace_count = 0
    #     brace_end = -1
    #     for i in range(brace_start, len(input_string)):
    #         if input_string[i] == '{':
    #             brace_count += 1
    #         elif input_string[i] == '}':
    #             brace_count -= 1
    #             if brace_count == 0:
    #                 brace_end = i
    #                 break
        
    #     if brace_end == -1:
    #         return {}
        
    #     dict_string = input_string[brace_start:brace_end+1]
    #     dict_string = re.sub(r',\s*}', '}', dict_string)
    #     dict_string = re.sub(r',(\s*[}\]])', r'\1', dict_string)
        
    #     try:
    #         return json.loads(dict_string)
    #     except json.JSONDecodeError:
    #         return {}
    

    # # def extract_prediction_values(self, input_string):
    # #     """
    # #     提取字符串内所有<s> ... </s>包裹的JSON字典，返回一个dict列表
    # #     """
    # #     pattern = r'<s>\s*(\{.*?\})\s*</s>'
    # #     matches = re.findall(pattern, input_string, re.DOTALL)
    # #     dicts = []
    # #     for dict_string in matches:
    # #         dict_string = re.sub(r',\s*}', '}', dict_string)
    # #         try:
    # #             dicts.append(json.loads(dict_string))
    # #         except json.JSONDecodeError:
    # #             dicts.append({})
    # #     return dicts

    # def extract_prediction_values(self, input_string):
    #     # Simply delegate to extract_dictionary which now handles all cases
    #     return self.extract_dictionary(input_string)
