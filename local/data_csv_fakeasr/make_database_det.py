import os
import json
from tqdm import tqdm
import pandas as pd
import mutagen

import argparse

from save_csv import save_csv


def main(DSETs, data_dir, save_dir):
    for dset in DSETs:
        labels = {'audio_id': [], 'audio_path':[], 'audio_len':[], 'realfake':[]}
        # load wav.scp, utt2dur, utt2cls, etc.
        utt2path=dict([line.strip().split() for line in open(os.path.join(data_dir, dset,  'wav.scp'))])
        utt2dur=dict([line.strip().split() for line in open(os.path.join(data_dir, dset,'utt2dur'))])
        utt2cls=dict([line.strip().split() for line in open(os.path.join(data_dir, dset,'utt2cls'))])
    
        for uttid, label in tqdm(utt2cls.items()):
            labels['audio_id'].append(uttid)
            labels['audio_path'].append(utt2path[uttid])
            labels['audio_len'].append(utt2dur[uttid])
            labels['realfake'].append(label)
    
        df = pd.DataFrame(labels)
        print(df.shape)
        database_name=os.path.basename(data_dir.rstrip('/'))
        save_csv(df, database_name, dset, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset into CSV")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the output CSV files")
    parser.add_argument('--DSETs', nargs='+', default=['train', 'dev', 'eval'], help="List of dataset splits to process")

    args = parser.parse_args()

    main(args.DSETs, args.data_dir, args.save_dir)

