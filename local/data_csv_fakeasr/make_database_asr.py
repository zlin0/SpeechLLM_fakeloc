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
    for word, start_time, end_time in timestamps:
        if is_fake(start_time, end_time, fake_periods):
            # If the word is located in the fake region, add [fake]
            updated_transcript.append(f"{word}[fake]")
        else:
            updated_transcript.append(word)

    return " ".join(updated_transcript)

def read_json_file(json_file_path, save_dir):
    """
    {
    "file": "test/000001.mp4",
    "n_fakes": 0,
    "fake_periods": [],
    "timestamps": [
      [
        "not",
        0.0,
        0.2
      ],
      ...
      [
        "life",
        3.8,
        4.0
      ]
    ],
    "duration": 4.224,
    "transcript": "not the point the point is that she died satisfied with the life",
    "original": null,
    "modify_video": false,
    "modify_audio": false,
    "split": "test",
    "video_frames": 103,
    "audio_channels": 1,
    "audio_frames": 65536
  },
    """


    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    records = []
    for av_info in tqdm(data):

        records.append({
            'audio_id': av_info['file'],
            'audio_path': "/export/fs05/arts/dataset/LAV-DF/LAV-DF/"+av_info['file'],
            'audio_len': av_info['duration'],
            'transcript': av_info['transcript'],
            'dset': av_info['split']
        })

    df = pd.DataFrame(records)
    print(df.shape)
    for dset in ["train", "dev", "test"]:
        # Create an explicit copy of the slice to avoid SettingWithCopyWarning
        save_csv(df[df['dset']==dset].copy(), "lav-df", dset, save_dir)



def main():
    parser = argparse.ArgumentParser(description="Process dataset into CSV")
    parser.add_argument('--json_file', type=str, default='/export/fs05/arts/dataset/LAV-DF/LAV-DF/metadata.json',
                        help="Path to the dataset directory")
    parser.add_argument('--save_dir', type=str, default='./data/',
                        help="Directory to save the output CSV files")
    args = parser.parse_args()

    read_json_file(args.json_file, args.save_dir)


if __name__ == "__main__":
    main()
