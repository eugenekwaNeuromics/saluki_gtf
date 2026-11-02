# Script for obtaining Saluki half-life predictions from GTFs

This script uses code and resources from the following projects:

* Basenji (Apache License 2.0): https://github.com/calico/basenji/blob/master/LICENSE 
* Saluki model weights (Creative Commons Attribution 4.0 International): https://zenodo.org/records/6326409

Please refer to the respective licences for terms of use, and cite the original paper:

> Agarwal, Vikram, and David R. Kelley. "The genetic and biochemical determinants of mRNA degradation rates in mammals." Genome biology 23.1 (2022): 245.

## How to use

This script was created with the purpose of obtaining transcript-level half-life values, to determine the functional consequences of novel UTR / exon usage etc. Decided to write this myself owing to some issues with model loading and parameter specification: see [this issue](https://github.com/ericmalekos/RNA-Zoo/issues/1) and [this](https://github.com/vagarwal87/saluki_paper/issues/15). 

### Step 1: Download the genome FASTA and annotation GTF files

The original publication uses mm10 for mouse data (and GRCh38 for human). 

```
wget ftp://ftp.ensembl.org/pub/release-90/gtf/mus_musculus/Mus_musculus.GRCm38.90.gtf.gz
wget ftp://ftp.ensembl.org/pub/release-90/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz
```

### Step 2: Download relevant `basenji` scripts and files 

`run_saluki.py` mainly requires:

* `basenji/basenji/rnann.py`: For building the model. This in turn depends on `layers.py` and `metrics.py` scripts in the same directory.
* `basenji/manuscripts/saluki/params.json`: Specifies the model architecture. This is also duplicated in the weights directory, i.e. `datasets/deeplearning/train_gru/params.json`.

```
git clone https://github.com/calico/basenji.git
```

### Step 3: Download the model's weights

In the resultant folder, the weights are stored in `datasets/deeplearning/train_gru/f*_c*/train/` subdirectories, where `f*_c*` specifies the folds and crosses used during training.
There should be 50 of such folders, each with their own `model*_best.h5` file, where `model0` specifies the weights for the model with the human head, and `model1` for the mouse head.

Accordingly, for each `f*_c*/` directory, `test0/preds.h5` specifies model predictions for human genes, and `test1/preds.h5` for mouse. The latter was used to obtain the values in `mouse_mm10_GenesWithSingleTranscripts.csv.gz`, after selecting genes with only a single known transcripts. (This filtering was necessary because for genes with multiple transcripts, I will not be able to compare them against the relevant `preds.h5`, which reports a single prediction per _gene_.)

```
wget https://zenodo.org/records/6326409/files/datasets.zip
```

### Step 4: Run `run_saluki.py`

To obtain the predictions in `preds_Saluki_mouse.csv.gz`:

```
conda run -n tensorflow_env python run_saluki.py \
    -o OUT/ \
    -b basenji/ \
    -w datasets/ \
    -i GENOME/Mus_musculus.GRCm38.90.gtf.gz \
    -a GENOME/Mus_musculus.GRCm38.90.gtf.gz \
    -f GENOME/Mus_musculus.GRCm38.dna.primary_assembly.fa \
    -s "mouse" \
    -g mouse_mm10_GenesWithSingleTranscripts.csv.gz \
)
```

where `tensorflow_env` is the user environment that supports `tensorflow`. Note the duplication of the same GTF file for both `-i` and `-a` flags. Briefly, the flags denote:

* `-o`: Output directory.
* `-b`: Cloned `basenji` directory.
* `-w`: Directory containing the weight data. Assumes that the user has directly unzipped the downloaded `datasets.zip` file.
* `-i`: Input GTF file. This will be used to specify exon usage for e.g. novel transcripts.
* `-a`: Annotated GTF file. This will be used to obtain CDS and alternative transcript names. In the example, `-i` and `-a` are the same file, because the same file holds both information sets.
* `-f`: Corresponding genomic FASTA file specifying sequences.
* `-s`: Chosen species. Can be `"human"`, `"mouse"`, or `"both"` (default).
* `-g`: Specifies which genes to obtain transcript-level half-life values for. Either individual gene IDs (e.g. `-g ENSGXXX -g ENSGYYY`) or a single `.csv` file with a `"gene"` column.
* `-p`: File prefix for saving output `preds_Saluki_{species}.csv.gz` files. Default is `""`. 

For additional details, kindly refer to the `run_saluki.py` script or run:

```
conda run -n tensorflow_env python run_saluki.py --help
```

The `correlation.png` image shows the correlation of `test1/preds.h5` values and those obtained by running `run_saluki.py`. Diagonal is $y = x$.

![Sanity check that script returns results that correspond to test data predictions.](./correlation.png)