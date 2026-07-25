import torch
from transformers import AutoProcessor, AutoFeatureExtractor

import torch
from torch.utils.data import Dataset, Sampler
import torchaudio
from torchtune.datasets import ConcatDataset
import pandas as pd
import random
import numpy as np
import json
import ast
import sys
import subprocess
import tempfile
import os 
import torchvision

def make_weighted_sampler_from_dataset(dataset, dtype=torch.double):
    """
    Build a WeightedRandomSampler from a CompositeAudioDataset.
    - If dataset contains only one sub-dataset: return None (use shuffle=True outside).
    - If multiple sub-datasets: expand dataset.datasets_weights into per-sample weights.
    
    Args:
        dataset: CompositeAudioDataset or similar object containing .dataset.datasets and .datasets_weights
        dtype: torch dtype for the weights tensor (default: torch.double)
    
    Returns:
        WeightedRandomSamfpler if multiple sub-datasets, otherwise None.
    """
    # Get the list of sub-datasets (in case of ConcatDataset)
    subs = getattr(getattr(dataset, "dataset", None), "datasets", None)

    # If not multiple sub-datasets, return None (then set shuffle=True)
    if not (isinstance(subs, (list, tuple)) and len(subs) > 1):
        return None

    sizes = [len(d) for d in subs]
    dataset_weights = getattr(dataset, "datasets_weights", None)
    if dataset_weights is None or len(dataset_weights) != len(sizes):
        raise ValueError("datasets_weights is missing or does not match the number of sub-datasets.")

    # Expand per-dataset weight to per-sample weight
    weights_per_sample = torch.cat([
        torch.full((sz,), float(w) / sz, dtype=dtype)
        for w, sz in zip(dataset_weights, sizes)
    ])
    assert len(weights_per_sample) == len(dataset), "weights length must match the total number of samples."

    return torch.utils.data.WeightedRandomSampler(weights_per_sample,
                                  num_samples=len(weights_per_sample),
                                  replacement=True)
                                  
class MyCollator:
    def __init__(self, audio_encoder_name, tokenizer, use_text=False):
        self.use_text = use_text
        self.audio_encoder_name = audio_encoder_name
        self.tokenizer = tokenizer
        if self.audio_encoder_name in ["facebook/hubert-xlarge-ll60k", 
                                        "microsoft/wavlm-large", 
                                        'microsoft/wavlm-base-plus']:
            self.hubert_processor = AutoFeatureExtractor.from_pretrained(audio_encoder_name) # change according to the encoder
        else:
            self.hubert_processor = None
            

    def __call__(self, batch):
        audio_ids, mel, pre_tokenized_ids, transcript_tokenized_ids, post_tokenized_ids, output_tokenized_ids = [], [], [], [], [], []
        frame_labels = []
        has_frame_labels = False
        
        for el_tuple in batch:
            # el_tuple is immutable, recreate it for sanitization
            sample_frame_labels = None
            if len(el_tuple) == 7:
                audio_id, waveform, pre, t, post, out, sample_frame_labels = el_tuple
                has_frame_labels = True
            elif len(el_tuple) == 6:
                audio_id, waveform, pre, t, post, out = el_tuple
            else:
                raise ValueError(f"Unexpected sample format with {len(el_tuple)} fields")
            
            # User request: if audio is empty, skip and output id.
            # We "skip" by nullifying the waveform, which will be handled in the trainer.
            if waveform is None or waveform.shape[1] <= 400:
                print(f"Skipping empty/short audio sample: {audio_id}")
                waveform = torch.empty(1, 0) # Nullify waveform
            
            # Create the element for processing
            el = (audio_id, waveform, pre, t, post, out)

            aid, m, pe, transc, po, o = self.process(el)

            # The original "empty found" check is removed as it was confusing.
            # The trainer will now be responsible for handling empty 'm' tensors.
            
            audio_ids.append(aid)
            mel.append(m)
            pre_tokenized_ids.append(pe)
            transcript_tokenized_ids.append(transc)
            post_tokenized_ids.append(po)
            output_tokenized_ids.append(o)
            if has_frame_labels:
                if sample_frame_labels is None:
                    sample_frame_labels = torch.empty(0, dtype=torch.long)
                elif not torch.is_tensor(sample_frame_labels):
                    sample_frame_labels = torch.tensor(sample_frame_labels, dtype=torch.long)
                else:
                    sample_frame_labels = sample_frame_labels.long()
                frame_labels.append(sample_frame_labels)
            
        base_output = (
            audio_ids,
            self.pad(mel), 
            self.pad(pre_tokenized_ids).long(), 
            self.pad(transcript_tokenized_ids).long(),
            self.pad(post_tokenized_ids).long(), 
            self.pad(output_tokenized_ids).long())

        if has_frame_labels:
            return base_output + (self.pad(frame_labels, pad_value=-100).long(),)

        return base_output

    def process(self, element):
        audio_id, waveform, pre_speech_prompt, transcript, post_speech_prompt, output_prompt = element

        if waveform is not None and self.hubert_processor is not None:
            # if "openai/whisper" in self.audio_encoder_name:
            #     mel = self.wav_2_mel(waveform).unsqueeze(0)
            # else:
            mel = self.hubert_processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values
        else:
            mel = torch.empty(1, 0, dtype=torch.float32)

        # Support audio-text attention alignment if enabled

        pre_tokenized_ids = self.tokenizer(pre_speech_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        
        if transcript is not None:
            transcript_to_tokenize = transcript
        else:
            transcript_to_tokenize = ""
            # print(f"transcript is None for {audio_id}")
        transcript_tokenized_ids = self.tokenizer(transcript_to_tokenize, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"][0]
        assert transcript_tokenized_ids.dim() >= 1, f"Invalid transcript token shape: {transcript_tokenized_ids.shape} for {audio_id}"
        
        post_tokenized_ids = self.tokenizer(post_speech_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"][0]
        output_tokenized_ids = self.tokenizer(self.tokenizer.bos_token + output_prompt + self.tokenizer.eos_token, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"][0]
        
        return audio_id, mel, pre_tokenized_ids, transcript_tokenized_ids, post_tokenized_ids, output_tokenized_ids

    # def pad(self, list_tensors):
    #     max_len = np.max([i.shape[1] for i in list_tensors])
    #     # print(list_tensors[0].shape)
    #     output = torch.zeros((len(list_tensors), max_len))

    #     for i,t in enumerate(list_tensors):
    #         squeezed_t = t.squeeze() #[1,1] -> ERROR
    #         if squeezed_t.dim() == 0:
    #             # Handle 0-d tensor (scalar) from a single-token sequence
    #             output[i, 0] = squeezed_t
    #         else:
    #             output[i, :len(squeezed_t)] = squeezed_t #mono channel only
    #     return output 

    def pad(self, list_tensors, pad_value=0):
        # list_tensors: we hope each element is 1-D [T]
        flat = []
        for t in list_tensors:
            if not torch.is_tensor(t):
                t = torch.tensor(t)
            flat.append(t.view(-1))                 # allow [1,T]/[T]/[] to be 1-D

        max_len = max(t.numel() for t in flat) if len(flat) > 0 else 0  #

        # dtype/device follow input (avoid float + CPU issues)
        dtype = flat[0].dtype if len(flat) > 0 else torch.long
        device = flat[0].device if len(flat) > 0 else "cpu"
        output = torch.full((len(flat), max_len), pad_value, dtype=dtype, device=device)

        for i, t in enumerate(flat):
            L = t.numel()
            if L > 0:
                output[i, :L] = t

        return output

def read_video(path: str):
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    video = video.permute(0, 3, 1, 2) / 255
    # audio = audio.permute(1, 0)
    if audio.shape[1] == 0:
        audio = torch.zeros(1, 1)
    return video, audio, info

class AudioDataset(Dataset):
    def __init__(self, csv_file, mode='train',random_keys_prob=0.001, max_len = -1, max_size=-1, fields=[], 
                 use_text=False, max_duration=-1, frame_binary_classification=False):
        self.data_frame = pd.read_csv(csv_file, low_memory=False)
        if max_size>0 and len(self.data_frame) > max_size : self.data_frame = self.data_frame.sample(n=max_size)
        self.data_frame = self.data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Filter out samples with audio longer than max_duration (in seconds)
        if max_duration > 0:
            original_size = len(self.data_frame)
            # Try 'duration' column first, then 'audio_len'
            if 'duration' in self.data_frame.columns:
                self.data_frame = self.data_frame[self.data_frame['duration'] <= max_duration]
            elif 'audio_len' in self.data_frame.columns:
                self.data_frame = self.data_frame[self.data_frame['audio_len'] <= max_duration]
            else:
                print(f"Warning: Neither 'duration' nor 'audio_len' column found in CSV. Cannot filter by max_duration={max_duration}s")
            
            filtered_size = len(self.data_frame)
            if filtered_size < original_size:
                print(f"Filtered out {original_size - filtered_size} samples with duration > {max_duration}s (remaining: {filtered_size})")
        
        self.mode = mode
        if len(fields)==0: self.labels = ['transcript', 'gender', 'emotion', 'age', 'accent', 'noises', 'summary'] #'isspeech', 
        else: self.labels = fields
        self.max_len = max_len*16_000
        self.random_keys_prob=random_keys_prob
        self.use_text=use_text
        self.max_duration = max_duration
        self.frame_binary_classification = frame_binary_classification
        self.failed_audio_log = 'failed_audio_samples.log'  # Log file for failed samples
        if self.use_text and 'transcript' in self.labels: print("Warning: You should not give transcripts and ask to predict them, that could be counter-productive.")
        # datasets = list(self.data_frame['dataset'])
        # dataset_to_index = {d:i for i,d in enumerate(set(datasets))}
        # dataset_indices = np.array([dataset_to_index[d] for d in datasets])
        # index_to_weight = [len(dataset_indices[dataset_indices==i]) for i in set(dataset_indices)]
        # self.datasets_weights = np.array([len(self.data_frame)/index_to_weight[i] for i in dataset_indices])
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # Load audio
        audio_row = self.data_frame.iloc[idx]
        audio_path = audio_row['audio_path']
        audio_id = audio_row['audio_id']
        # if self.mode=='test':print(idx, audio_row, audio_path, sep='\t')
        try:
            if pd.isna(audio_path):
                waveform = None
            elif '.mp3' in audio_path:
                waveform, sample_rate = torchaudio.load(audio_path, format='mp3')
            elif '.mp4' in audio_path and 'AV-Deepfake1M-PlusPlus' in audio_path:
                waveform, sample_rate = read_video(audio_path)[1], 16000 
                # read_video returns (video, audio, info), we take audio and assume 16kHz for AVdeepfake1M++
            else:
                waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            try:
                waveform, sample_rate = self._load_with_ffmpeg(audio_path)
                print(f"   ffmpeg fallback succeeded: {audio_path}")
            except Exception as ffmpeg_error:
                print(f"  Failed to load audio: {audio_path} (audio_id={audio_id})")
                print(f"   Error: {e}")
                print(f"   ffmpeg fallback error: {ffmpeg_error}")
                # Log failed sample and return empty waveform to skip
                self._log_failed_sample(audio_id, audio_path, str(ffmpeg_error))
                waveform = None
        
        # Skip processing if waveform is None (failed to load)
        if waveform is None:
            if self.frame_binary_classification:
                return audio_id, None, {}, "", "", torch.empty(0, dtype=torch.long)
            return audio_id, None, {}, "", ""

        if waveform.shape[0]==2:waveform=torch.mean(waveform, axis=0).unsqueeze(0)
        duration_sec = waveform.shape[1] / 16000.0
        clip_start_sec = 0.0
        
        # Additional runtime check: skip samples exceeding max_duration (as fallback)
        if self.max_duration > 0:
            if duration_sec > self.max_duration:
                print(f"Skipping sample {audio_id}: duration {duration_sec:.2f}s exceeds max_duration {self.max_duration}s")
                if self.frame_binary_classification:
                    return audio_id, None, {}, "", "", torch.empty(0, dtype=torch.long)
                return audio_id, None, {}, "", ""
        
        if waveform.shape[1]>self.max_len and self.max_len>0: 
            start = int(np.random.rand(1)*(waveform.shape[1]-self.max_len))
            clip_start_sec = start / 16000.0
            waveform=waveform[:, start:start+self.max_len]
            duration_sec = waveform.shape[1] / 16000.0
            print(f"DEBUG: shape after truncate: {waveform.shape}")
        # # Prepare labels dictionary based on mode and probability
        labels_str = {}
        if self.mode == 'train' and random.random() < self.random_keys_prob:
            random_labels = random.sample(self.labels, k=random.randint(1, len(self.labels)))
            for label in random_labels:
                if label in audio_row and pd.notnull(audio_row[label]):
                    formatted_label = label.capitalize()
                    if audio_row[label] == True or audio_row[label] == False:
                        labels_str[formatted_label] = audio_row[label]
                    else:
                        labels_str[formatted_label] = str(audio_row[label]).lower()
        else:
            # include all available labels
            for label in self.labels:
                if label in audio_row and pd.notnull(audio_row[label]):
                    formatted_label = label.capitalize()
                    if audio_row[label] == True or audio_row[label] == False:
                        labels_str[formatted_label] = audio_row[label]
                    else:
                        labels_str[formatted_label] = str(audio_row[label]).lower()

        if self.use_text and 'transcript' in audio_row:
            transcript = audio_row['transcript']
            if transcript is None: print(f"transcript is empty for {audio_id}")
        elif self.use_text and 'asr_transcript' in audio_row:
            transcript = audio_row['asr_transcript']
        else:
            transcript = ""

        if 'context' in audio_row.index:
            conv_history = audio_row['context']
        else:
            conv_history = ""
        
        if(self.frame_binary_classification):
            audio_fake_period = audio_row['fake_periods'] if 'fake_periods' in audio_row else None
            # Build frame labels on the effective clip timeline (supports random truncation).
            frame_labels = self._extract_frame_labels(
                audio_fake_period,
                duration_sec,
                clip_start_sec=clip_start_sec,
            )
            return audio_id, waveform, labels_str, conv_history, transcript, frame_labels
        else:
            return audio_id, waveform, labels_str, conv_history, transcript

    def _extract_frame_labels(self, audio_fake_period, duration_sec, frame_len=0.02, clip_start_sec=0.0):
        """
        Convert fake periods (absolute timeline in seconds) to frame-level binary labels
        on the current clip timeline.

        Args:
            audio_fake_period: fake intervals in absolute audio time, e.g. [[s, e], ...]
            duration_sec: duration of current clip in seconds
            frame_len: frame size in seconds
            clip_start_sec: absolute start time of current clip in original audio
        """
        total_frames = int(duration_sec / frame_len)
        frame_labels = torch.zeros(total_frames, dtype=torch.long)

        if audio_fake_period is None:
            return frame_labels
        if isinstance(audio_fake_period, float) and np.isnan(audio_fake_period):
            return frame_labels
        if isinstance(audio_fake_period, str):
            text = audio_fake_period.strip()
            if text == "" or text.lower() == "nan":
                return frame_labels
            try:
                audio_fake_period = ast.literal_eval(text)
            except Exception:
                return frame_labels
        if not isinstance(audio_fake_period, (list, tuple)):
            return frame_labels

        clip_end_sec = clip_start_sec + duration_sec

        for period in audio_fake_period:
            if not isinstance(period, (list, tuple)) or len(period) < 2:
                continue
            start_time, end_time = period

            # Intersect fake period with current clip range.
            overlap_start = max(float(start_time), float(clip_start_sec))
            overlap_end = min(float(end_time), float(clip_end_sec))
            if overlap_end <= overlap_start:
                continue

            # Convert to clip-local time before frame index conversion.
            local_start = overlap_start - clip_start_sec
            local_end = overlap_end - clip_start_sec

            start_frame = int(local_start / frame_len)
            end_frame = int(local_end / frame_len)
            start_frame = max(0, min(start_frame, total_frames))
            end_frame = max(0, min(end_frame, total_frames))
            frame_labels[start_frame:end_frame] = 1  # Mark frames in the fake period as 1
        return frame_labels

    def _log_failed_sample(self, audio_id, audio_path, error_msg):
        """Log failed audio samples to a file for later inspection/filtering."""
        try:
            with open(self.failed_audio_log, 'a', encoding='utf-8') as f:
                f.write(f"{audio_id}\t{audio_path}\t{error_msg}\n")
        except Exception as log_error:
            print(f"  Failed to log error: {log_error}")

    def _load_with_ffmpeg(self, audio_path, sample_rate=16000):
        """
        Fallback loader using ffmpeg CLI to extract audio from media files.
        """
        tmp_file = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_file.close()
            cmd = [
                "ffmpeg",
                "-y",
                "-v", "error",
                "-i", audio_path,
                "-vn",
                "-ac", "1",
                "-ar", str(sample_rate),
                "-f", "wav",
                tmp_file.name
            ]
            result = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "ffmpeg failed without stderr")
            waveform, sr = torchaudio.load(tmp_file.name)
            return waveform, sr
        finally:
            if tmp_file and os.path.exists(tmp_file.name):
                try:
                    os.remove(tmp_file.name)
                except Exception:
                    pass
    
class InstructionalAudioDataset(AudioDataset):
    def __init__(self, csv_file, mode='train',random_keys_prob=0.001,  max_len = -1, max_size=-1, fields=[], 
                 use_text=False, prob_text=0.5, audio_transcript_align=None, max_duration=-1, use_audio=True,
                 frame_binary_classification=False, test_on=None, icl_num_shots=0, icl_train_csv=None, icl_seed=42):
        """
        Initialize the class with the specified CSV file, mode, and random keys probability.

        Args:
            csv_file (str): The path to the CSV file.
            mode (str, optional): The mode of the operation, defaults to 'train'.
            max_duration (float, optional): Maximum audio duration in seconds. Samples longer than this will be filtered. Default -1 (no limit).
        Returns:
            None
        """
        self.use_text = use_text
        self.prob_text = prob_text
        self.audio_transcript_align = audio_transcript_align
        self.use_audio = use_audio
        self.frame_binary_classification = frame_binary_classification
        self.test_on = test_on if test_on is not None else ('AT' if use_text and use_audio else 'T' if use_text else 'A')
        super().__init__(csv_file, mode, random_keys_prob=random_keys_prob, max_len = max_len, max_size=max_size, 
                         fields=fields, use_text=use_text, max_duration=max_duration, frame_binary_classification=frame_binary_classification)
        self.icl_num_shots = int(icl_num_shots)
        self.icl_train_csv = icl_train_csv
        self.icl_seed = int(icl_seed)
        self.icl_prompt_prefix = self._build_icl_prompt_prefix()
        if('fake_transcript' in fields and 'fake_periods' in fields):
            self.instruction_phrases = [
                # —— Joint task: transcription with [fake] tags + region detection
                "Transcribe and tag [fake] words; also return synthetic regions as [[start, end], ...] in seconds. Output strict JSON.",
                "You will be given an audio clip of speech. It may be machine-generated or human-spoken. Predict the synthetic regions in seconds as [[start_time, end_time], ...]. Use 2–3 decimal places, merge adjacent/overlapping segments, return [] if none, and provide the transcript with [fake] appended to suspected words.",
                "Here is a speech sample. Determine whether it is human or machine-generated. Transcribe it and append [fake] after suspected synthetic words. Also return the synthetic time regions in seconds as [[start, end], ...]. Example: please unlock[fake] my account.",

                # —— Region-only task ——
                "Evaluate the provided speech audio and detect segments that sound machine-generated. Output a list of intervals in seconds as [[start, end], ...]. If none, return [].",
                "Identify likely synthetic segments and return a strictly formatted JSON array of intervals in seconds: [[start, end], ...]. Constraints: start < end, 0 ≤ start < audio_duration, 0 < end ≤ audio_duration, sorted ascending, non-overlapping.",
                "Detect synthetic regions in the audio and return [[start, end], ...] in seconds. Return [] if no suspicion.",

                # —— Word-only task ——
                "Transcribe the speech and tag synthetic words with [fake]. Example: please unlock[fake] my account.",
                "Transcribe the provided speech. If a word is likely synthetic, mark it by appending [fake]; otherwise leave it unmarked. Example: please unlock[fake] my account.",
                "You are asked to evaluate an audio recording. While transcribing, whenever you encounter a word that may have been generated by a machine, tag it with [fake]. Example: please unlock[fake] my account.",

                # —— Unified decision task (classification + regions + word tags) ——
                "Decide if the audio contains any synthetic speech. If yes, return the list of synthetic regions in seconds [[start, end], ...] and a transcript with [fake] after suspected words. Use strict JSON."
         ]
        elif('fake_transcript' in fields):
            if self.use_text and not self.use_audio:
                self.instruction_phrases = [
                    "Here is a transcript of speech. This text may have been machine-generated or spoken by a real person. Please determine which it is, and locate the words that are likely synthetic. Mark suspected fake words by appending [fake] after them. Example: please unlock[fake] my account.",
                ]
            elif self.use_text and (self.audio_transcript_align == "concat" or self.audio_transcript_align == "attention"):
                self.instruction_phrases =  [
                "Here is a speech sample with its transcription. This audio may have been machine-generated or spoken by a real person. Please determine which it is, and locate the words that are likely synthetic. Mark suspected fake words by appending [fake] after them. Example: please unlock [fake] my account.",
                "Listen to the given audio clip. It might be a real human voice or a synthetic one. Its transcription is also provided. Identify any words that sound machine-generated and label them with [fake] after the word. Example: please unlock [fake] my account.",
                "Here is a speech sample with its transcription. It could be spoken by a real human or generated by a machine. Please transcribe the audio, and after each word that seems synthetic, add [fake]. Example: please unlock [fake] my account.",
                "Tag synthetic words with [fake]. Example: please unlock[fake] my account.",
                "You are asked to evaluate an audio recording with its transcription. Determine if it is natural or synthetic. While transcribing, whenever you encounter a word that may have been generated by a machine, tag it with [fake]. Example: please unlock [fake] my account."
            ]
            else:
                self.instruction_phrases = [
                    "Here is a speech sample. This audio may have been machine-generated or spoken by a real person. Please determine which it is, and locate the words that are likely synthetic. Mark suspected fake words by appending [fake] after them. Example: please unlock[fake] my account.",
                    "Listen to the given audio clip. It might be a real human voice or a synthetic one. Identify any words that sound machine-generated and label them with [fake] after the word. Example: please unlock[fake] my account.",
                    "Here is a speech sample. It could be spoken by a real human or generated by a machine. Please transcribe the audio, and after each word that seems synthetic, add [fake]. Example: please unlock[fake] my account.",
                    "Transcribe the speech and tag synthetic words with [fake]. Example: please unlock[fake] my account.",
                    "Transcribe the provided speech. If you believe a word is synthetic, mark it by appending [fake]. Otherwise, leave it unmarked. Example: please unlock[fake] my account.",
                    "You are asked to evaluate an audio recording. Determine if it is natural or synthetic. While transcribing, whenever you encounter a word that may have been generated by a machine, tag it with [fake]. Example: please unlock[fake] my account."
                ]
        else:
            self.instruction_phrases = [
                "Is this audio real (bonafide/genuine) that spoken by human, or fake (spoof) that generated/manipulated by machine",
            ]
        # self.instruction_phrases = [
        #     "Is this audio real (bonafide/genuine) that spoken by human, or fake (spoof) that generated/manipulated by machine",
        # ]
    #### Get in-context learning prompt example:
    # For simplicity, we use the same CSV file as the main 
    # set for sampling ICL examples. In practice, a separate curated set of high-quality examples may be preferable.
    def _resolve_csv_path(self, csv_path):
        if not csv_path:
            return None
        candidate_paths = [csv_path]
        if not os.path.isabs(csv_path):
            candidate_paths.append(os.path.join("data", csv_path))
            if not csv_path.endswith(".csv"):
                candidate_paths.append(os.path.join("data", f"{csv_path}.csv"))
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                return candidate
        return None

    def _row_to_input_text(self, row, transcript_key="fake_transcript"):
        # transcript_candidates = ["transcript", "ori_transcript", "asr_transcript", "fake_transcript"]
        transcript_candidates = [transcript_key]
        transcript_text = ""
        for candidate_key in transcript_candidates:
            if candidate_key in row.index and pd.notnull(row[candidate_key]):
                transcript_text = str(row[candidate_key])
                if transcript_text.strip():
                    break

        # Keep speech tags in AT/A examples as textual placeholders.
        # trainer.encode() still inserts only one real speech embedding block for the current sample.
        if self.test_on == "T":
            if transcript_text:
                return f"Input:\n<transcript>{transcript_text}</transcript>"
            return "Input:\n<transcript></transcript>"
        #TODO: besides prompt, we may want to extract real acousti.
        if self.test_on == "AT":
            if transcript_text:
                return f"Input:\n<speech></speech>\n<transcript>{transcript_text}</transcript>"
            return "Input:\n<speech></speech>\n<transcript></transcript>"

        return "Input:\n<speech></speech>"

    def _row_to_labels_json(self, row):
        labels_str = {}
        for label in self.labels:
            if label not in row.index or pd.isna(row[label]):
                continue
            formatted_label = label.capitalize()
            value = row[label]
            if value is True or value is False:
                labels_str[formatted_label] = value
            else:
                labels_str[formatted_label] = str(value).lower()

        json_obj = {}
        for key, value in labels_str.items():
            if key == "Isspeech":
                key = "SpeechActivity"
            json_obj[key] = value
        return json_obj

    def _build_icl_prompt_prefix(self):
        if self.icl_num_shots <= 0 or not self.icl_train_csv:
            return ""

        resolved_csv = self._resolve_csv_path(self.icl_train_csv)
        if resolved_csv is None:
            print(f"Warning: ICL train CSV not found: {self.icl_train_csv}")
            return ""

        demo_frame = pd.read_csv(resolved_csv, low_memory=False)
        if len(demo_frame) == 0:
            return ""

        sample_size = min(self.icl_num_shots, len(demo_frame))
        sampled_frame = demo_frame.sample(n=sample_size, random_state=self.icl_seed).reset_index(drop=True)

        demo_blocks = []
        for demo_idx, (_, demo_row) in enumerate(sampled_frame.iterrows(), start=1):
            json_obj = self._row_to_labels_json(demo_row)
            if not json_obj:
                continue

            demo_input_text = self._row_to_input_text(demo_row, transcript_key='transcript')
            demo_text = (
                f"Example {demo_idx}:\n"
                f"{demo_input_text}\n\n"
                f"Output:\n{json.dumps(json_obj, ensure_ascii=False)}"
            )
            demo_blocks.append(demo_text)

        if not demo_blocks:
            return ""

        return "In-context examples:\n" + "\n\n".join(demo_blocks) + "\n\nNow solve the next example:\n\n"

    def _get_prompt_input_marker(self):
        if self.test_on == "T":
            return "<transcript>"
        return "<speech>"
    #### Finish in-context learning prompt example - above.
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if len(sample) == 6:
            audio_id, waveform, labels_str, conv_history, transcript, frame_labels = sample
        elif len(sample) == 5:
            audio_id, waveform, labels_str, conv_history, transcript = sample
            frame_labels = None
        else:
            raise ValueError(f"Unexpected sample format from AudioDataset: {len(sample)} fields")
        instruction_phrase = random.choice(self.instruction_phrases)
        input_marker = self._get_prompt_input_marker()

        pre_speech_prompt = self.icl_prompt_prefix + f"Instruction:\n{instruction_phrase} - ["
        pre_speech_prompt += ', '.join(['IsSpeech' if k == 'isSpeech' else k for k in labels_str.keys()]) + f"]\n\nInput:\n{input_marker}"
        pre_speech_prompt = pre_speech_prompt.replace("Isspeech", "SpeechActivity")
        if self.use_text and (self.test_on == "T" or random.random() <= self.prob_text):
            if self.test_on == "T":
                post_speech_prompt = f"</transcript>\n\n" + "Output:\n"
            elif(self.audio_transcript_align == "attention"):
                post_speech_prompt = f"</speech>\n\n" + "Output:\n"
            else:
                post_speech_prompt = f"</speech>\n\n<transcript>{transcript}</transcript>\n\n" + \
                    "Output:\n"
        else:
            post_speech_prompt = f"</speech>\n\n" + \
                "Output:\n"
            transcript = "" # Transcript not used in prompt, so clear it
        
        # Build a proper JSON object for the target, letting json.dumps handle all escaping
        json_obj = {}
        for key, value in labels_str.items():
            # Normalize key name
            if key == "Isspeech":
                key = "SpeechActivity"
            json_obj[key] = value

        # Use ensure_ascii=False so text stays readable; json.dumps will escape quotes correctly
        output_prompt = json.dumps(json_obj, ensure_ascii=False)
        if (self.frame_binary_classification):
            return audio_id, waveform, pre_speech_prompt, transcript, post_speech_prompt, output_prompt, frame_labels
        else:
            return audio_id, waveform, pre_speech_prompt, transcript, post_speech_prompt, output_prompt

class CompositeAudioDataset(Dataset):
    def __init__(self, list_of_datasets, mode='train', random_keys_prob=0.001, max_len = -1, max_size=-1, use_text=False, 
                 prob_text=0.5, audio_transcript_align=None, max_duration=-1, use_audio=True,
                 frame_binary_classification=False):
        datasets = []
        for data_name in list_of_datasets:
            data = InstructionalAudioDataset(
                        csv_file = f'./data/{data_name}.csv',
                        mode=mode, 
                        random_keys_prob=random_keys_prob,
                        max_len=max_len,
                        max_size=max_size,
                        fields=list_of_datasets[data_name],
                        use_text=use_text,
                        prob_text=prob_text,
                        audio_transcript_align=audio_transcript_align,
                        max_duration=max_duration,
                        use_audio=use_audio,
                        frame_binary_classification=frame_binary_classification
                        )
            datasets.append(data)
            print(f"Loaded {data_name}, length = {len(data)}")
            
        
        # if only one dataset, use it directly
        if len(datasets) == 1:
            self.dataset = datasets[0]
            self.len = len(self.dataset)
            self.datasets_weights = np.array([1.0])  # single dataset, weight is 1
        else:
            # more than one dataset, use ConcatDataset
            self.dataset = ConcatDataset(datasets)
            self.len = len(self.dataset)
            self.datasets_weights = np.array([self.len/len(d) for d in datasets])


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.dataset[idx]


# Example usage
if __name__ == "__main__": #debug
    dataset = InstructionalAudioDataset(csv_file='dev.csv', mode='test')
    audio_id, waveform, pre_speech_prompt, transcript, post_speech_prompt, output_prompt = dataset[121]

    print("--- Example Item ---")
    print(f"Audio ID: {audio_id}")
    print("\n--- Pre-speech prompt ---")
    print(pre_speech_prompt)
    print("\n--- Transcript ---")
    print(transcript)
    print("\n--- Post-speech prompt ---")
    print(post_speech_prompt)
    print("\n--- Output prompt (target) ---")
    print(output_prompt)
    print("\n--- Waveform ---")
    print(waveform)

