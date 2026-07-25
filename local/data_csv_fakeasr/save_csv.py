import pandas as pd

def save_csv(df, dataset, dset, save_dir):
    df['dataset'] = dataset
    df['set'] = dset
    assert 'audio_path' in df.columns, f"audio_path not found in {dataset} {set}"
    assert 'audio_len' in df.columns, f"audio_len not found in {dataset} {set}"

    save_path=f"{save_dir}/{dataset}_{dset}.csv"
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_csv(save_path, index=False, header=True)
    print(f"{dataset} {dset} saved! total length: {df.shape}")
    print(save_path)
