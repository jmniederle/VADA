import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.utils.normalization import med_mad, normalize_signal
from src.utils import get_project_root, pickle_save
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import os
import itertools


def pad_batch(batch):
    bases, raw = zip(*batch)
    padded_bases = pad_sequence(bases, batch_first=True, padding_value=0)
    padded_raw = pad_sequence(raw, batch_first=True, padding_value=0.0)
    return padded_bases, padded_raw

    # return pad_sequence(batch, batch_first=True, padding_value=0)


def pad_batch_kmer(batch):
    """
    To be used as collate_fn in dataloader for NanoKMerDataset class
    """
    raw, kmers = zip(*batch)
    padded_raw = pad_sequence(raw, batch_first=True, padding_value=0.0)
    padded_kmers = pad_sequence(kmers, batch_first=True, padding_value=0)
    return padded_kmers, padded_raw


def pad_batch_hdf5_kmer(batch):
    kmers, raw = zip(*batch)
    kmers, raw = torch.tensor(np.stack(kmers)), torch.tensor(np.stack(raw))
    non_pad_idx = get_right_most_non_zero_idx(raw)
    min_r_pad_idx = torch.max(non_pad_idx).item() + 1  # Find the min padded right-most padded index
    return kmers[:, :min_r_pad_idx], raw[:, :min_r_pad_idx]


def pad_batch_hdf5_kmer_no_reshape(batch):
    kmers, raw = zip(*batch)
    kmers, raw = torch.tensor(np.stack(kmers)), torch.tensor(np.stack(raw))
    return kmers, raw


def get_right_most_non_zero_idx(in_tensor):
    # Find indices of the rightmost non-zero elements in each row
    indices = torch.arange(in_tensor.shape[1])
    row_mask = (in_tensor != 0)
    rightmost_non_zero_indices = torch.argmax(row_mask * indices, dim=1)
    return rightmost_non_zero_indices


def clip_nano_read(nano_read):
    """
    Clip a Nanopore raw signal to only include part of the read that is aligned to the reference genome.
    Args:
        nano_read: ReadData object with raw signal to clip

    Returns: clipped raw signal

    """
    dna_bases_start_in_raw = nano_read.start_rel_to_raw
    dna_bases_end_in_raw = dna_bases_start_in_raw + nano_read.segmentation['start'][-1] \
                           + nano_read.segmentation['length'][-1]
    return nano_read.raw[dna_bases_start_in_raw:dna_bases_end_in_raw]


def split_array_by_length(a, split_len=100):
    """
    Split an array into subarrays of length split_len
    Args:
        a: array to split
        split_len: integer length of subarrays

    Returns: list of subarrays
    """
    n_splits = np.ceil(len(a) / split_len)
    return np.array_split(a, indices_or_sections=n_splits)


def process_bases(bases_arr):
    enc = OneHotEncoder(categories=[["A", "C", "T", "G"]], sparse_output=False)
    return enc.fit_transform(bases_arr.reshape(-1, 1))


def restore_dna_sequence(bases_one_hot):
    """
    Transform an array of one-hot encoded bases into an array of DNA string representations

    Args:
        bases_one_hot: 2D array of bases shape: [sequence_length, 4]

    Returns:

    """
    assert (len(bases_one_hot.shape) == 2) and (bases_one_hot.shape[-1] == 4), \
        "Invalid shape of base array, must have shape [sequence length, 4]"

    if not type(bases_one_hot) == np.ndarray:
        bases_one_hot = bases_one_hot.numpy()

    categories = np.array(["A", "C", "T", "G"])

    bases_one_hot = bases_one_hot[~(bases_one_hot == 0).all(axis=1)]  # Remove padded trailing zeroes

    return categories[np.argmax(bases_one_hot, axis=-1)]


def restore_kmer_sequence(kmers_encoded):
    """

    Args:
        kmers_encoded: ordinally encoded kmers, shape [batch_size, seq_len]

    Returns:

    """
    if not type(kmers_encoded) == np.ndarray:
        kmers_encoded = kmers_encoded.numpy()

    all_kmers = ["".join(k) for k in list(itertools.product("ACGT", repeat=5))]
    all_kmers.insert(0, "nan")
    all_kmers = np.array(all_kmers)
    return np.take(all_kmers, kmers_encoded)


def process_segmentation_split(split_segmentation, start_idx_split):
    """
    Given a window of a segmentation, corresponding to the split in raw signal, process the segmentation such that it is
    relative to the start index of the split and does not contain a negative first index.

    Args:
        split_segmentation: part of a segmentation
        start_idx_split: starting index of the nanopore split

    Returns:

    """
    segm_rel_to_start = split_segmentation['start'].astype(int) - start_idx_split

    # Find negative indices, caused by start of first DNA base being in previous window
    neg_idx = np.where(segm_rel_to_start < 0)

    if len(neg_idx) > 1:
        raise ValueError("Detected more than 1 negative idx in segmentation, this should not be possible.")

    segm_rel_to_start[neg_idx] = 0
    return segm_rel_to_start


def split_and_process_nano_read(nano_read, split_len, return_segmentation=False, normalize=True):
    """
    Split a Nanopore read into parts of maximum size split_len and split the DNA bases corresponding to each part of the
    raw signal.
    Args:
        normalize:
        return_segmentation:
        nano_read: ReadData object to split
        split_len: integer length of the splits

    Returns: list of tuples with raw nanopore read of max size split_len and corresponding DNA bases

    """
    raw_signal = clip_nano_read(nano_read)
    raw_indices = np.arange(len(raw_signal))
    split_indices = split_array_by_length(raw_indices, split_len=split_len)
    dna_bases = nano_read.segmentation

    raw_base_pairs = []
    carry_over = None
    for idx_arr in split_indices:
        bases_in_window = dna_bases[np.isin(dna_bases['start'], idx_arr)]

        # Check if we need to insert the last base of the previous slice into the current slice
        if carry_over is not None:
            bases_in_window = np.insert(bases_in_window, 0, carry_over)

        # Check if the last base of the current index slice overlaps with the next one, if so save the last base of the
        # current slice for the next slice
        if (bases_in_window['start'][-1] + bases_in_window['length'][-1]) > idx_arr[-1]:
            carry_over = bases_in_window[-1]

        else:
            carry_over = None

        processed_bases = process_bases(bases_in_window['base'])
        bases = torch.tensor(processed_bases)
        raw = raw_signal[idx_arr]

        if normalize:
            med, mad = med_mad(raw, factor=1.0)  # Calculate signal median and median absolute deviation
            raw = normalize_signal(raw, med, mad)

        raw = torch.tensor(raw)

        if return_segmentation:
            segm_processed = process_segmentation_split(bases_in_window, idx_arr[0])
            segm = torch.tensor(segm_processed)
            raw_base_pairs.append((bases, raw, segm))

        else:
            raw_base_pairs.append((bases, raw))

    return raw_base_pairs


def get_kmer_one_hot_encoder():
    all_kmers = ["".join(k) for k in list(itertools.product("ACGT", repeat=5))]
    all_kmers.insert(0, "nan")
    return OneHotEncoder(categories=[all_kmers], sparse_output=False)


def split_and_process_nano_read_kmer(nano_read, split_len, kmer_one_hot_enc, normalize=True):
    """
    Split a Nanopore read into parts of maximum size split_len and split the DNA bases corresponding to each part of the
    raw signal, returning a categorically encoded k-mer
    Args:
        kmer_one_hot_enc:
        normalize:
        nano_read: ReadData object to split
        split_len: integer length of the splits

    Returns: list of tuples with raw nanopore read of max size split_len and corresponding DNA bases

    """
    raw_signal = clip_nano_read(nano_read)
    raw_indices = np.arange(len(raw_signal))
    split_indices = split_array_by_length(raw_indices, split_len=split_len)
    dna_bases = nano_read.segmentation

    raw_base_pairs = []
    carry_over = None
    for idx_arr in split_indices:
        bases_in_window = dna_bases[np.isin(dna_bases['start'], idx_arr)]

        # Check if we need to insert the last base of the previous slice into the current slice
        if carry_over is not None:
            bases_in_window = np.insert(bases_in_window, 0, carry_over)

        # Check if the last base of the current index slice overlaps with the next one, if so save the last base of the
        # current slice for the next slice
        if (bases_in_window['start'][-1] + bases_in_window['length'][-1]) > idx_arr[-1]:
            carry_over = bases_in_window[-1]

        else:
            carry_over = None

        # get the segmentations
        segm_processed = process_segmentation_split(bases_in_window, idx_arr[0])

        raw = raw_signal[idx_arr]

        if normalize:
            med, mad = med_mad(raw, factor=1.0)  # Calculate signal median and median absolute deviation
            raw = normalize_signal(raw, med, mad)

        # Make array with number of repeats of each base in raw signal where [3, 8, ...] indicates the first base is
        # repeated 3 times, the second 8 times etc.
        base_repeats = np.diff(segm_processed, append=len(raw))

        # Creat an idx array that matches length of raw signal, indicating at each raw timepoint the current base
        base_idx_in_raw = np.repeat(np.arange(len(segm_processed)), base_repeats)

        # Create an array with only the base indexes of complete kmers, i.e. the first two bases in the sequence do not
        # correspond to a full k-mer as there is not yet a full kmer of data in the raw split
        first_full_kmer, last_full_kmer = min(base_idx_in_raw) + 2, max(base_idx_in_raw) - 2
        base_idx_complete_kmers = \
            base_idx_in_raw[np.where((base_idx_in_raw >= first_full_kmer) & (base_idx_in_raw <= last_full_kmer))]

        # Index the base array using the base_idx_in_raw, the result is an array of shape [5, split_len - ...], where
        # the first dimension contains the relevant k-mer at timepoint in the raw signal
        kmers = bases_in_window['base'][
            [base_idx_complete_kmers - 2, base_idx_complete_kmers - 1, base_idx_complete_kmers,
             base_idx_complete_kmers + 1, base_idx_complete_kmers + 2]]

        # join the base strings into one kmer, shape is now [split_len - ...]
        kmers = np.char.add(np.char.add(np.char.add(np.char.add(kmers[0, :], kmers[1, :]), kmers[2, :]), kmers[3, :]),
                            kmers[4, :])

        # Check if this split does not contain any full kmers, if this is the case, insert nan's for the whole split
        if first_full_kmer > last_full_kmer:
            kmers = np.insert(kmers, 0, np.repeat("nan", len(base_idx_in_raw)))

        # If there are full kmers, insert the needed nan's at the start and end for the incomplete kmers
        else:
            kmers = np.insert(kmers, 0, np.repeat("nan", sum((base_idx_in_raw < first_full_kmer))))
            kmers = np.append(kmers, np.repeat("nan", sum((base_idx_in_raw > last_full_kmer))))

        kmers_one_hot = kmer_one_hot_enc.fit_transform(kmers.reshape(-1, 1))
        kmers_categorical = kmers_one_hot.argmax(axis=1)
        kmers_categorical = torch.tensor(kmers_categorical)

        if kmers_categorical.shape[0] > split_len:
            print("kmer sequence too large, something's up")

        raw = torch.tensor(raw)

        # segm = torch.tensor(segm_processed)
        raw_base_pairs.append((raw, kmers_categorical))

    return raw_base_pairs


def create_small_dataset(dataset, max_items, save_dir):
    data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=6)
    save_dir = os.path.join(get_project_root(), save_dir)
    samples = []
    print("Creating small dataset.")
    for i, (bases, raw) in tqdm(enumerate(data_loader), total=max_items):
        if i > max_items:
            break
        samples.append((bases.squeeze(0).numpy(), raw.squeeze(0).numpy()))
    pickle_save(samples, save_dir)


if __name__ == "__main__":
    print(os.getcwd())
    # read_data = read = next(iter(read_fast5('../data/0a0bdc5c-8f8f-41ea-a4d1-4ff6344fac3e.fast5').values()))
    # res = split_and_process_nano_read_kmer(read_data, 500, kmer_one_hot_enc=get_kmer_one_hot_encoder(), normalize=True)

    a = restore_dna_sequence(torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]))

    print("done")
