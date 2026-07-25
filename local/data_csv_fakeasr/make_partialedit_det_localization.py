import os
import json
import difflib
from tqdm import tqdm
import pandas as pd
import argparse
import jiwer

from save_csv import save_csv

def is_fake(timestamp_start, timestamp_end, fake_periods):
    # check fake_periods to find overlaped fake region
    for fake_start, fake_end in fake_periods:
        if timestamp_start < fake_end and timestamp_end > fake_start:
            return True
    return False

def label_fake_words(origin, modified, all=False):
    # same as the one used for llamapartialspoof
    """
    Label fake words by comparing origin and modified transcripts using ASR alignment.
    Mark words as [fake] if they are insertions or substitutions.
    
    Args:
        origin: original/reference transcript
        modified: modified/hypothesis transcript  
        all: if True, mark all words as [fake]
    
    Returns:
        modified transcript with [fake] tags on inserted/substituted words
    """
    
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),         # lower case
        jiwer.RemovePunctuation(),   # delete pronaction 
        jiwer.RemoveMultipleSpaces(), # delete reduncant spaces
        jiwer.Strip()                 # delete spaces at the begining and end.
    ])

    # Treat hyphens as word separators so "Four-leaf" -> "Four leaf"
    origin_clean = transform(origin.replace('-', ' ').strip())
    modified_clean = transform(modified.replace('-', ' ').strip())
    
    ori_tokens = origin_clean.strip().split()
    mod_tokens = modified_clean.strip().split()

    if all:
        out = [word + "[fake]" for word in mod_tokens]
        return ' '.join(out)
    
    if not origin or not modified:
        return modified
    
    # Use difflib to get alignment opcodes (match, replace, insert, delete)
    matcher = difflib.SequenceMatcher(None, ori_tokens, mod_tokens)
    opcodes = matcher.get_opcodes()
    
    # Build output with [fake] tags for insertions and substitutions
    out = []
    for op, i1, i2, j1, j2 in opcodes:
        if op == 'equal':
            # Words match - keep as is
            out.extend(mod_tokens[j1:j2])
        elif op == 'replace':
            # Substitution - mark as [fake]
            for token in mod_tokens[j1:j2]:
                out.append(f"{token}[fake]")
        elif op == 'insert':
            # Insertion - mark as [fake]
            for token in mod_tokens[j1:j2]:
                out.append(f"{token}[fake]")
        elif op == 'delete':
            # Deletion - skip (these are in origin but not in modified)
            pass
    
    return ' '.join(out)
    

def check_audio_len(uttid, audio_len, fake_periods):
    """
    Check if any fake periods exceed the audio length.
    Return True if there is an error, False otherwise.
    """
    if(audio_len <= 0):
        print(f"Warning: {uttid} has non-positive audio length {audio_len}.")
        return True

    for start, end in fake_periods:
        if start >= audio_len or end > audio_len:
            print(f"Warning: {uttid} has audio length {audio_len} but fake periods {fake_periods} may exceed audio length.")
            return True

def read_metadata_file(wav_scp, txt_scp, metadata_file, save_dir):
    """
    Input: 
    1. wav.scp: <uttid> <wav_path>
       e.g. E1/p229/p229_001_edited <wav_path>
    2. txt file scp: <uttid> <txt_path>
    3. metadata file:
        - `filename`: The name of the audio file.
        - `start of the edited region (s)`: The starting time (in seconds) of the first edited region.
        - `end of the edited region (s)`: The ending time (in seconds) of the first edited region.
        [- `start` and `end may appear more than once.]
        - `total duration (s)`: The total duration (in seconds) of the audio file.
    note that:    
    txt may contain quote and ","

    Output:
    csv file with row:
    # audio_id,audio_path,audio_len,realfake,ori_transcript,transcript,fake_transcript,fake_periods,dset,dataset,fakeword_num,word_tolnum,set
    # ori_transcript is the original transcript without any [fake] tags.
    # transcript is the modified transcript without any [fake] tags.
    # fake_transcript is the modified transcript with [fake] tags on modified words.
    """

    # Read wav.scp: mapping uttid -> wav_path
    utt2wav = {line.strip().split()[0]: line.strip().split()[1] for line in open(wav_scp, 'r', encoding='utf-8')}
    # Read txt scp: mapping short_uttid -> txt_path
    uttid2txtpath = {line.strip().split()[0]: line.strip().split()[1] for line in open(txt_scp, 'r', encoding='utf-8')}
    utt2dur_file = wav_scp.replace('wav.scp', 'utt2dur') 
    utt2dur = {line.strip().split()[0]: float(line.strip().split()[1]) for line in open(utt2dur_file, 'r', encoding='utf-8')}

    dset = os.path.basename(os.path.dirname(wav_scp)).split('-')[0] # e.g. train, dev, test

    # Read metadata file (comma separated) to get fake period # partialedit only provided this for fake.
    fake_periods_dict = {}
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split(',')
            uttid = arr[0].replace('.wav', '') # remove .wav suffix to match uttid in wav.scp
            times = [float(x) for x in arr[1:] if x]
            fake_periods = []
            for i in range(0, len(times)-1, 2):
                fake_periods.append([times[i], times[i+1]])
            fake_periods_dict[uttid] = fake_periods

    csv_rows = []
    for uttid, audio_path in tqdm(utt2wav.items()):
        if uttid not in fake_periods_dict:
            fake_periods_dict[uttid] = []  # if no fake periods, set to empty list

        txt_path = uttid2txtpath.get(uttid, '')
        audio_len = utt2dur.get(uttid, 0.0)
        error = check_audio_len(uttid, audio_len, fake_periods_dict[uttid])

        if txt_path and os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
        else: # can not find txt.
            # raise ValueError(f"Cannot find txt for {uttid} at {txt_path}")
            print(f"Cannot find txt for {uttid} at {txt_path}")
            continue

            
        # read original transcript
        if(uttid.find("edited") >= 0): #fake
            realfake = 'fake' 
            ori_uttid=uttid.split('/')[-1].replace('_edited', '')
            oritxt_path = uttid2txtpath.get(ori_uttid, '')
            if oritxt_path and os.path.exists(oritxt_path):
                with open(oritxt_path, 'r', encoding='utf-8') as f:
                    ori_transcript = f.read().strip()
            else: # can not find txt.
                raise ValueError(f"Cannot find original txt for {ori_uttid} at {oritxt_path}")

            fake_transcript = label_fake_words(ori_transcript, transcript) if transcript else ""
        else: #real
            realfake = 'real' 
            ori_transcript = transcript
            fake_transcript = transcript

        # deal with fake_periods: convert to string for csv, e.g. [[s1,e1],[s2,e2]
        fake_periods = fake_periods_dict.get(uttid, [])
        if fake_periods:
            fake_periods_str = "[" + ",".join([f"[{s},{e}]" for s, e in fake_periods]) + "]"
        else:
            fake_periods_str = "[]"
        fake_word_num = fake_transcript.count('[fake]')
        word_tolnum = len(fake_transcript.strip().split())
        if(fake_word_num == word_tolnum and uttid.find("-partial") != -1):
            print("debug full fake:", uttid)
        
        if not error:
            csv_rows.append({
                'audio_id': uttid,
                'audio_path': audio_path,
                'audio_len': audio_len,
                'realfake': realfake,
                'ori_transcript': ori_transcript,
                'transcript': transcript,
                'fake_transcript': fake_transcript,
                'fake_periods': fake_periods_str,
                'dset': dset,
                'dataset': 'partialedit',
                'fakeword_num': fake_transcript.count('[fake]'),
                'word_tolnum': len(fake_transcript.split()),
                'set': dset
            })

    # header = ['audio_id', 'audio_path', 'audio_len', 'realfake',
            #   'transcript', 'faketranscript', 'fake_periods', 'dset', 'dataset', 'set']
    df = pd.DataFrame(csv_rows)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f'partialedit_{dset}.csv'), index=False)
    print(f"partialedit {dset} saved to {save_dir}! total length: {df.shape}")


def main():
    parser = argparse.ArgumentParser(description="Process dataset into CSV")
    parser.add_argument('--wav_scp', type=str, default='/export/fs05/lzhan268/workspace/wedefense/egs/localization/partialeditv2/v15_ssl_mhfa/data/partialeditv2/E1E2_vctk/train/wav.scp',
                        help="Path to the dataset data folder")
    parser.add_argument('--txt_scp', type=str, default='/export/fs05/lzhan268/workspace/wedefense/egs/localization/partialeditv2/v15_ssl_mhfa/data/partialeditv2/E1E2_vctk/train/txt.scp',
                        help="Path to the dataset directory")
    parser.add_argument('--metadata_file', type=str, default='/export/fs05/arts/dataset/PartialEdit_v2/PartialEdit_E1E2_v2.csv',
                        help="Path to the dataset directory")

    parser.add_argument('--save_dir', type=str, default='/export/fs05/lzhan268/workspace/03llm/SpeechLLM_fakeloc/data/partialeditv2/E1E2_vctk',
                        help="Directory to save the output CSV files")
    args = parser.parse_args()

    read_metadata_file(args.wav_scp, args.txt_scp, args.metadata_file, args.save_dir)




if __name__ == "__main__":
    main() # with pre-defined paramaters for partialspoof.
    # debug_label_fake()
    # main()
