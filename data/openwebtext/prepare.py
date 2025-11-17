
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# num of workers in .map() call
# good number is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding('gpt2')

if __name__ == '__main__':
    # takes 54GB in hf .cache dir, about 8M documents (8,013,769)
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # why create test set here ?
        # probably to test ability of pretrain model
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename test split to val

    # >>> split_dataset
    # DatasetDic({
    #   train: Dataset({
    #       features: ['text'],
    #       num_rows: 8009762
    #   })
    #   val: Dataset({
    #       features: ['text'],
    #       num_rows: 4007
    #   })
    # })

    # now tokenize the dataset, first define encoding functions (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        totoal_batches = 1024
    
        idx = 0
        for batch_idx in tqdm(range(totoal_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=totoal_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

