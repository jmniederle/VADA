from src.utils import load_vada_checkpoint
from src.datasets.data_util import restore_kmer_sequence
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os


def plot_real_generated_results(raw_nano, sample_nano, kmer_aligned=None, xlim=(200, 399.4)):
    fig, axs = plt.subplots(1, 2, figsize=(15, 4), tight_layout=True)
    axs[0].plot(raw_nano)
    axs[1].plot(sample_nano, c='tab:orange')

    kmer_change = (kmer_aligned != np.roll(kmer_aligned, 1))
    x_change = np.where(kmer_change)
    kmer_at_change = kmer_aligned[x_change]

    for pos_x, k in zip(x_change[0], kmer_at_change):
        axs[0].annotate(k, xy=(pos_x,-3.8), rotation=90)
        axs[0].vlines(pos_x, ymin=-3, ymax=3, colors='tab:gray', linestyles="dotted")

    axs[0].set_xlim(xlim)
    axs[1].set_xlim(xlim)
    axs[0].set_ylim((-4, 3))
    axs[1].set_ylim((-4, 3))
    fig.suptitle("Real (left) and generated (right) nanopore observations")
    fig.show()
    fig.savefig("demo_fig.png", dpi=350)


def demo(run_id=0):
    # # Load model
    # if run_id <= 4:
    #     config_path = f'src/configs/config_VADA_no_aux_{run_id}.json'

    # elif run_id == 5:
    #     config_path = 'src/configs/config_VADA_with_aux.json'
    configs_base_path = 'src/configs/'
    configs = sorted(os.listdir(configs_base_path))
    config_path = os.path.join(configs_base_path, configs[run_id])

    model = load_vada_checkpoint(config_path)

    # Load demo batch
    kmers, raw = np.load("kmer_batch.npy"), np.load("raw_batch.npy")

    kmers, raw = torch.tensor(kmers), torch.tensor(raw)

    # loss computation
    loss, reconstr_loss, dkl_zy_loss, aux_loss = model.compute_loss_parallel(raw, kmers)

    # sampling
    raw_samples = model.sample(kmers[:, :])

    # plot
    plot_idx = np.random.randint(len(raw_samples))

    kmer_seq = restore_kmer_sequence(kmers[plot_idx])
    plot_real_generated_results(raw[plot_idx].detach(), raw_samples[plot_idx].detach(), kmer_aligned=kmer_seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run number")
    parser.add_argument('--run', type=int, choices=range(16), default=0,
                        help="An integer in range [0, 5] with a default of 0")

    args = parser.parse_args()
    demo(run_id=args.run)
