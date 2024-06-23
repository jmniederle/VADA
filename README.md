# VADA

Code for the paper: "VADA: a Data-Driven Simulator for Nanopore Sequencing"

Create conda environment using:
```
conda create --name vada_env --file vada_requirements.txt
```

## Demo
To see an example of loss computation, sampling and generate an example plot, run: 

```
python VADA_demo.py 
```

To select a different run choose one of 0-4 for each VADA training run, or 5 for a training run of VADA trained with
the auxiliary regressor. For example:

```
python VADA_demo.py --run 3
```

## Data
The data that was used for training VADA is publicly available, to download follow instructions on [GitHub Repo](https://github.com/nanoporetech/bonito/).
*Note: this download is ~30GB*

For training on data where the reference DNA sequence has not been aligned with the nanopore observations, use the [Tombo Package](https://nanoporetech.github.io/tombo/tombo.html)

To read a sequence of nanopore observations use the `read_fast5()` function from `src/utils/read.py`. And to preprocess 
a sequence of nanopore observations use `split_and_process_nano_read_kmer()` in `datasets/data.util.py`, where arguments 
should be specified as follows:
- `nano_read`: the ReadData object (output of `read_fast5()`)
- `split_len`: the length of subsequences to split the nanopore sequence into
- `kmer_one_hot_enc`: a kmer onehotencoder object, i.e. by running `get_kmer_one_hot_encoder()` from `datasets/data_util.py`
- `normalize`: whether to normalize the sequences


## Training
The model was trained using configurations that can be found in `configs/config_VADA_training.json`
