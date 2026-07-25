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
    
def read_metadata_file(wav_scp, real_txt, fake_txt, metadata_file, save_dir):
    """
    Input: 
    1. wav.scp: <uttid> <wav_path>
       e.g. dev01-cosyvoice-full_8842_304647_000046_000000 <wav_path>
    2. real_txt file: <short_uttid> <ori_txt> 
       e.g., 8842_304647_000046_000000 A "PERPLEXING" ART CRITIC
       1272_128104_000003_000003 To many, no doubt
    3. fake_txt file: <short_uttid> <mod_txt> (row: real_txt > fake_txt )
    4. metadata file: <uttid> <utterance-duration> <utterance-label> <segment1> <segment2> ... <segmentN>
        Each segment is formatted as follows:
        <start>-<end>-<label>
        e.g., dev01-yourtts-partial-cf_8842_304647_000017_000002 5.0540 spoof 0.0000-0.5930-bonafide 0.5930-2.4630-spoof 2.4630-3.3890-bonafide 3.3890-3.7490-spoof 3.7490-3.7950-bonafide 3.7950-5.0540-spoof
    
    note that:    
    subset always be test for llamapartialspoof
    txt may contain quote and ","

    Output:
    csv file with row:
    # audio_id,audio_path,audio_len,realfake,transcript,fake_transcript,fake_periods,dset,dataset,set
    # dev01-cosyvoice-full_8842_304647_000046_000000,/path/in/wav.scp,4.16,real,"A 'PERPLEXING' ART CRITIC",,train,lav-df,train
    # dev/004056.mp4,/export/fs05/arts/dataset/LAV-DF/LAV-DF/dev/004056.mp4,7.936,fake,a lot of litigation luckily but if anybody does their research they'll see that these these betrayals did in fact,a lot of litigation luckily[fake] but if anybody does their research they'll see that these these betrayals did in fact,"[[0.9, 1.844]]",dev,lav-df,dev

    """

    # Read wav.scp: mapping uttid -> wav_path
    utt2path = {line.strip().split()[0]: line.strip().split()[1] for line in open(wav_scp, 'r', encoding='utf-8')}

    # Read real_txt: mapping short_uttid -> ori_txt
    shortuttid2txt = {}
    with open(real_txt, 'r', encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split(None, 1)
            if len(arr) == 2:
                shortuttid, ori_txt = arr
                shortuttid2txt[shortuttid] = ori_txt

    # Read fake_txt: mapping short_uttid -> mod_txt
    shortuttid2modtxt = {}
    with open(fake_txt, 'r', encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split(None, 1)
            if len(arr) == 2:
                shortuttid, mod_txt = arr
                shortuttid2modtxt[shortuttid] = mod_txt

    del shortuttid

    # Read metadata and compile the csv rows
    csv_rows = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            uttid, duration, global_label = parts[:3]
            segments = parts[3:]
            # Example uttid: dev01-cosyvoice-full_8842_304647_000046_000000
            # Pull short_uttid from end
            if uttid.find('-') != -1:
                short_uttid = "_".join(uttid.split('_')[-4:])  # dynamic based on the sample, fallback below works for 5-part ids
            else:
                short_uttid = uttid

            # Parse segment regions for fake periods
            fake_periods = []
            for seg in segments:
                try:
                    start, end, label = seg.split('-')
                    if label.strip().lower() == 'spoof':
                        fake_periods.append( (float(start), float(end)) )
                except Exception:
                    continue

            # Prepare fields
            audio_path = utt2path.get(uttid, '')
            audio_len = duration
            dset = "test"
            dataset = "llamapartialspoof"
            set_field = "test"

            # TODO here or mapping.
            # Transcripts: original and fake
            # # asr_transcript: best estimate of true; fakeasr_transcript: always "fake" version if available
            # # For "fake" versions, add [fake] after the modified words. We'll append [fake] after each word that's different.
            ori_txt = shortuttid2txt.get(short_uttid, "")
            mod_txt = shortuttid2modtxt.get(short_uttid, "")
            if(uttid == 'dev01-ljjets-partial-cf_5895_34615_000014_000000'):
                print("debugging")
            if(uttid.find("-clean") == -1): #fake
                if (uttid.find("-partial-") != -1): #partial
                    transcript = mod_txt
                    fake_transcript = label_fake_words(ori_txt, mod_txt) if mod_txt else ""
                else: #full
                    transcript = mod_txt
                    fake_transcript = label_fake_words(ori_txt, mod_txt, all=True) if mod_txt else ""
                # fake_transcript = label_fake_words(ori_txt, mod_txt) if mod_txt else ""
            else: # real
                transcript = ori_txt
                fake_transcript = ori_txt
            # if short_uttid in shortuttid2txt else ""
            # To generate the one with [fake] label.

            # fake_periods CSV field
            # Format fake_periods as string: example '[[0.9, 1.844]]'
            if fake_periods:
                fake_periods_str = "[" + ",".join([f"[{s},{e}]" for s, e in fake_periods]) + "]"
            else:
                fake_periods_str = "[]"
            fake_word_num = fake_transcript.count('[fake]')
            word_tolnum = len(fake_transcript.strip().split())
            if(fake_word_num == word_tolnum and uttid.find("-partial") != -1):
                print("debug full fake:", uttid)

            row = {
                'audio_id': uttid,
                'audio_path': audio_path,
                'audio_len': audio_len,
                'realfake': global_label,
                'ori_transcript': ori_txt,
                'transcript': transcript,
                'fake_transcript': fake_transcript,
                'fake_periods': fake_periods_str,
                'dset': dset,
                'dataset': dataset,
                'fakeword_num': fake_word_num,
                'word_tolnum': word_tolnum,
                'set': set_field
            }
            csv_rows.append(row)

    # header = ['audio_id', 'audio_path', 'audio_len', 'realfake',
            #   'transcript', 'faketranscript', 'fake_periods', 'dset', 'dataset', 'set']
    df = pd.DataFrame(csv_rows)
    print(df.shape)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df.to_csv(f"{save_dir}/llamapartialspoof.csv", index=False, header=True)
    print(f"{dataset} {dset} saved to {save_dir}! total length: {df.shape}")



def main():
    parser = argparse.ArgumentParser(description="Process dataset into CSV")
    parser.add_argument('--wav_scp', type=str, default='/export/fs05/lzhan268/workspace/wedefense/egs/localization/llamapartialspoof/data/llamapartialspoof/0a/wav.scp',
                        help="Path to the dataset data folder")
    parser.add_argument('--ori_txt_file', type=str, default='/export/fs05/lzhan268/workspace/01partialspoof/LlamaPartialSpoof/transcripts/real.txt',
                        help="Path to the dataset directory")
    parser.add_argument('--mod_txt_file', type=str, default='/export/fs05/lzhan268/workspace/01partialspoof/LlamaPartialSpoof/transcripts/fake.txt',
                        help="Path to the dataset directory")
    parser.add_argument('--metadata_file', type=str, default='/export/fs05/arts/dataset/LlamaPartialSpoof/label_R01TTS.0.a.txt',
                        help="Path to the dataset directory")
    parser.add_argument('--save_dir', type=str, default='/export/fs05/lzhan268/workspace/03llm/SpeechLLM_fakeloc/data',
                        help="Directory to save the output CSV files")
    args = parser.parse_args()

    read_metadata_file(args.wav_scp, args.ori_txt_file, args.mod_txt_file, args.metadata_file, args.save_dir)


def debug():
    parser = argparse.ArgumentParser(description="Process dataset into CSV")
    parser.add_argument('--wav_scp', type=str, default='/export/fs05/lzhan268/workspace/wedefense/egs/localization/llamapartialspoof/data/llamapartialspoof/0a/wav.scp',
                        help="Path to the dataset data folder")
    parser.add_argument('--ori_txt_file', type=str, default='/export/fs05/lzhan268/workspace/01partialspoof/LlamaPartialSpoof/transcripts/real.txt',
                        help="Path to the dataset directory")
    parser.add_argument('--mod_txt_file', type=str, default='/export/fs05/lzhan268/workspace/01partialspoof/LlamaPartialSpoof/transcripts/fake.txt',
                        help="Path to the dataset directory")
    parser.add_argument('--metadata_file', type=str, default='/export/fs05/arts/dataset/LlamaPartialSpoof/label_R01TTS.0.a.txt',
                        help="Path to the dataset directory")
    parser.add_argument('--save_dir', type=str, default='/export/fs05/lzhan268/workspace/03llm/SpeechLLM_fakeloc/data/llamapartialspoof_new',
                        help="Directory to save the output CSV files")
    args = parser.parse_args()

    read_metadata_file(args.wav_scp, args.ori_txt_file, args.mod_txt_file, args.metadata_file, args.save_dir)




def debug_label_fake():
    real_txt="/export/fs05/lzhan268/workspace/01partialspoof/LlamaPartialSpoof/transcripts/real.txt"
    fake_txt="/export/fs05/lzhan268/workspace/01partialspoof/LlamaPartialSpoof/transcripts/fake.txt"
    shortuttid2txt = {}
    with open(real_txt, 'r', encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split(None, 1)
            if len(arr) == 2:
                shortuttid, ori_txt = arr
                shortuttid2txt[shortuttid] = ori_txt

    # Read fake_txt: mapping short_uttid -> mod_txt
    shortuttid2modtxt = {}
    with open(fake_txt, 'r', encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split(None, 1)
            if len(arr) == 2:
                shortuttid, mod_txt = arr
                shortuttid2modtxt[shortuttid] = mod_txt
    del shortuttid

    # shortuttid="7850_281318_000029_000002"
    debug_list=['6241_66616_000009_000001', '6345_93306_000005_000000', '5694_64025_000005_000003', '3170_137482_000045_000001', '5338_284437_000020_000000'];
    fullid_list=['dev01-gptsovits-full_6241_66616_000009_000001', 'dev01-gptsovits-full_6345_93306_000005_000000', 'dev01-gptsovits-full_5694_64025_000005_000003', 'dev01-gptsovits-full_3170_137482_000045_000001', 'dev01-gptsovits-full_5338_284437_000020_000000']
    fullid_list=['dev01-gptsovits-full_5338_284437_000020_000000']

    for uttid in fullid_list:
        # print(shortuttid)
        # print(shortuttid2txt[shortuttid])
        # print(shortuttid2modtxt[shortuttid])
        if uttid.find('-') != -1:
            short_uttid = "_".join(uttid.split('_')[-4:])  # dynamic based on the sample, fallback below works for 5-part ids
        else:
            short_uttid = uttid
        ori_txt = shortuttid2txt[short_uttid]
        mod_txt = shortuttid2modtxt[short_uttid]
        
        if(uttid.find("-clean") == -1): #fake
            if (uttid.find("-partial-") != -1): #partial
                transcript = mod_txt
                fake_transcript = label_fake_words(ori_txt, mod_txt) if mod_txt else ""
            else: #full
                transcript = mod_txt
                fake_transcript = label_fake_words(ori_txt, mod_txt, all=True) if mod_txt else ""
            # fake_transcript = label_fake_words(ori_txt, mod_txt) if mod_txt else ""
        else: # real
            transcript = ori_txt
            fake_transcript = ori_txt
            
        print(transcript, " | ", fake_transcript)


# dev01-gptsovits-full_6241_66616_000009_000001,/export/fs05/arts/dataset/LlamaPartialSpoof/R01TTS.0.a/dev01-gptsovits-full_6241_66616_000009_000001.wav,4.9400,
# spoof,Minnetoki herself replied to this tennis racket.,
# Minnetoki[fake] herself[fake] replied[fake] to[fake] this[fake] tennis[fake] racket.[fake],"[[0.0,4.94]]",test,llamapartialspoof,test
# dev01-gptsovits-full_6345_93306_000005_000000,/export/fs05/arts/dataset/LlamaPartialSpoof/R01TTS.0.a/dev01-gptsovits-full_6345_93306_000005_000000.wav,7.9400,spoof,"She poured her tea, moved her journal closer, uncapped her fountain pen, and scribbled her theme on the empty page.","She[fake] poured[fake] her[fake] tea,[fake] moved[fake] her[fake] journal[fake] closer,[fake] uncapped[fake] her[fake] fountain[fake] pen,[fake] and[fake] scribbled[fake] her[fake] theme[fake] on[fake] the[fake] empty[fake] page.[fake]","[[0.0,7.94]]",test,llamapartialspoof,test
# dev01-gptsovits-full_5694_64025_000005_000003,/export/fs05/arts/dataset/LlamaPartialSpoof/R01TTS.0.a/dev01-gptsovits-full_5694_64025_000005_000003.wav,7.1200,spoof,The air was full of balls and deadly missiles. The her corps was carrying off the dying and wounded soldier.,The[fake] air[fake] was[fake] full[fake] of[fake] balls[fake] and[fake] deadly[fake] missiles.[fake] The[fake] her[fake] corps[fake] was[fake] carrying[fake] off[fake] the[fake] dying[fake] and[fake] wounded[fake] soldier.[fake],"[[0.0,7.12]]",test,llamapartialspoof,test
# dev01-gptsovits-full_3170_137482_000045_000001,/export/fs05/arts/dataset/LlamaPartialSpoof/R01TTS.0.a/dev01-gptsovits-full_3170_137482_000045_000001.wav,9.8200,spoof,"Their fascination misleading me, I prattled like an amateur expert, I bloviated, I cited sources I had never actually read.","Their[fake] fascination[fake] misleading[fake] me,[fake] I[fake] prattled[fake] like[fake] an[fake] amateur[fake] expert,[fake] I[fake] bloviated,[fake] I[fake] cited[fake] sources[fake] I[fake] had[fake] never[fake] actually[fake] read.[fake]","[[0.0,9.82]]",test,llamapartialspoof,test
# dev01-gptsovits-full_5338_284437_000020_000000,/export/fs05/arts/dataset/LlamaPartialSpoof/R01TTS.0.a/dev01-gptsovits-full_5338_284437_000020_000000.wav,3.3000,spoof,"""No, your Majesty,"" she replied; ""I'm only ----""","""No,[fake] your[fake] Majesty,""[fake] she[fake] replied;[fake] ""I'm[fake] only[fake] ----""[fake]","[[0.0,3.3]]",test,llamapartialspoof,test


if __name__ == "__main__":
    debug() # with pre-defined paramaters for partialspoof.
    # debug_label_fake()
    # main()
