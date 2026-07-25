import json
from tqdm import tqdm
import pandas as pd
import argparse

from save_csv import save_csv

def is_fake(timestamp_start, timestamp_end, fake_periods):
    # check fake_periods to find overlaped fake region
    for fake_start, fake_end in fake_periods:
        if timestamp_start < fake_end and timestamp_end > fake_start:
            return True
    return False

def create_updated_transcript(timestamps, fake_periods):
    """Create an updated transcript string based on fake regions.
    e.g., This is a real word -> This is a real[fake] word"""
    updated_transcript = []  

    # iterate all word in timestamps 
    for item in timestamps:
        word, start_time, end_time = item.values()
        if is_fake(start_time, end_time, fake_periods):
            # If the word is located in the fake region, add [fake]
            updated_transcript.append(f"{word}[fake]")
        else:
            updated_transcript.append(word)

    return " ".join(updated_transcript)

def read_json_file(metadata_scp, save_dir):
    """
  {
    "file": "id00012/21Uxsk56VDQ/00001/fake_video_real_audio.mp4",
    "original": "id00012/21Uxsk56VDQ/00001/real.mp4",
    "split": "train",
    "modify_type": "visual_modified",
    "audio_model": null,
    "fake_segments": [
      [
        3.28,
        3.44
      ]
    ],
    "audio_fake_segments": [],
    "visual_fake_segments": [
      [
        3.28,
        3.44
      ]
    ],
    "video_frames": 236,
    "audio_frames": 151552
  },
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    utt2metadata=dict([line.split() for line in open(metadata_scp)])
    
    
    records = []
    missing_transcripts = []  # 存储缺失 transcripts 的样本ID
    problematic_transcripts = []  # 存储有问题的 transcripts 格式
    
    for uttid, metadata in tqdm(utt2metadata.items()):
        try:
            with open(metadata, 'r') as json_file:
                av_info = json.load(json_file)
        except Exception as e:
            problematic_transcripts.append({
                'uttid': uttid,
                'path': metadata,
                'issue': f'Error in JSON: {str(e)}'
            })
            continue
        
        # Check if 'transcripts' key exists
        if 'transcripts' not in av_info:
            missing_transcripts.append(metadata)
            continue
        
        transcripts = av_info['transcripts']
        
        # 检查 transcripts 格式是否异常
        is_valid = True
        if isinstance(transcripts, str):
            is_valid = False
            problematic_transcripts.append({
                'uttid': uttid,
                'path': metadata,
                'issue': f'transcripts 是字符串而非列表'
            })
        elif isinstance(transcripts, list):
            if len(transcripts) > 0:
                first_item = transcripts[0]
                if isinstance(first_item, str):
                    is_valid = False
                    problematic_transcripts.append({
                        'uttid': uttid,
                        'path': metadata,
                        'issue': f'transcripts 是字符串列表'
                    })
                elif isinstance(first_item, dict):
                    if 'word' not in first_item:
                        is_valid = False
                        problematic_transcripts.append({
                            'uttid': uttid,
                            'path': metadata,
                            'issue': f'transcripts 字典缺失 word 字段: {list(first_item.keys())}'
                        })
        
        if not is_valid:
            continue
        
        # Bug fix: Use 'not' for boolean negation, not '~'.
        real_or_fake = 'fake' if len(av_info['audio_fake_segments'])>0 else 'real'
        
        # Bug fix: Call the updated function which returns the new transcript string.
        if(real_or_fake == 'fake'):
            try:
                new_transcript = create_updated_transcript(av_info.get("transcripts", []), av_info.get("fake_segments", []))
            except Exception as e:
                problematic_transcripts.append({
                    'uttid': uttid,
                    'path': metadata,
                    'issue': f'处理 fake transcript 时出错: {str(e)}'
                })
                continue
            fake_periods = av_info.get('fake_segments', '')
        else:
            try:
                new_transcript = " ".join([w["word"] for w in av_info['transcripts']])
            except Exception as e:
                problematic_transcripts.append({
                    'uttid': uttid,
                    'path': metadata,
                    'issue': f'处理 real transcript 时出错: {str(e)}'
                })
                continue
            
            fake_periods = ''
            
        if "avdeepfake1m" in metadata_scp or "av-deepfake1m" in metadata_scp:   
            wav_dir=f"/export/fs06/arts/dataset/AV-Deepfake1M/{av_info['split']}/{av_info['file']}"
        elif "avdeepfake1m++" in metadata_scp or "av-deepfake1m++" in metadata_scp or "avdeepfake1mPP" in metadata_scp:
            wav_dir=f"/export/fs05/arts/dataset/AV-Deepfake1M-PlusPlus/{av_info['split']}/{av_info['file']}"
        else:
            raise ValueError("Unknown dataset in metadata_scp path.")
        
        records.append({
            'audio_id': av_info['file'],
            'audio_path': wav_dir,
            'audio_len': av_info['audio_frames']/16000.0,
            'realfake': real_or_fake,
            'transcript': " ".join([w["word"] for w in av_info['transcripts']]),
            'fake_transcript': new_transcript,
            'fake_periods': fake_periods,
            'fakeword_num': new_transcript.count('[fake]'),
            'word_tolnum': len(av_info['transcripts']),
            'dset': av_info['split']
        })

    df = pd.DataFrame(records)
    print(df.shape)
    #only one split for all input:
    if "avdeepfake1m" in metadata_scp or "av-deepfake1m" in metadata_scp:
        dset = av_info['split']
    else:
        dset = 'train' if "/train/" in metadata_scp else 'val' if '/val/' in metadata_scp else 'unknown' or None
    # 保存缺失 transcripts 的样本ID到文件
    if missing_transcripts:
        missing_file=f"{save_dir}/{dset}_missing_transcripts.txt"
        with open(missing_file, 'w') as f:
            for utt in missing_transcripts:
                f.write(f"{utt}\n")
        print(f"\n总共有 {len(missing_transcripts)} 个样本缺失 'transcripts' 键")
        print(f"缺失样本ID已保存到: {missing_file}")
    
    # 保存有问题的 transcripts 到文件
    if problematic_transcripts:
        problematic_file=f"{save_dir}/{dset}_problematic_transcripts.txt"
        with open(problematic_file, 'w') as f:
            f.write(f"总共找到 {len(problematic_transcripts)} 个有问题的样本\n\n")
            for item in problematic_transcripts:
                f.write(f"UttID: {item['uttid']}\n")
                f.write(f"Path: {item['path']}\n")
                f.write(f"Issue: {item['issue']}\n")
                f.write("-"*80 + "\n")
        print(f"\n总共有 {len(problematic_transcripts)} 个样本的 transcripts 格式异常")
        print(f"有问题的样本ID已保存到: {problematic_file}")
    
    print(f"\n✓ 成功处理 {len(records)} 个有效样本")
    print(f"  - 缺失 transcripts: {len(missing_transcripts)}")
    print(f"  - 格式异常: {len(problematic_transcripts)}")
    

    if "avdeepfake1m++" in metadata_scp or "av-deepfake1m++" in metadata_scp:
        save_csv(df.copy(), "avdeepfake1m++", dset, save_dir)
    elif "avdeepfake1m" in metadata_scp or "av-deepfake1m" in metadata_scp:
        save_csv(df[df['dset']==dset].copy(), "avdeepfake1m", dset, save_dir)


def main():
    parser = argparse.ArgumentParser(description="Process dataset into CSV")
    # parser.add_argument('--json_scp', type=str, default='/export/fs05/lzhan268/workspace/wedefense/egs/localization/avdeepfake1m/v15_ssl_mhfa/data/avdeepfake1m/train_audioonly_realvideo/metadata.scp',
    #                     help="Path to the scp for json files.")
    # parser.add_argument('--save_dir', type=str, default='./data/av-deepfake1m-audioonly_realvideo/debug',
    #                     help="Directory to save the output CSV files")
    parser.add_argument('--json_scp', type=str, default='/export/fs05/lzhan268/workspace/wedefense/egs/localization/avdeepfake1m/v15_ssl_mhfa/data/avdeepfake1m/train_audioonly_realvideo/metadata.scp',
                        help="Path to the scp for json files.")
    parser.add_argument('--save_dir', type=str, default='./data/av-deepfake1m-audioonly_realvideo/debug',
                        help="Directory to save the output CSV files")
    args = parser.parse_args()

    read_json_file(args.json_scp, args.save_dir)


if __name__ == "__main__":
    main()
