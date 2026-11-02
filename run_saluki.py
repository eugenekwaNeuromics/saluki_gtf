## This script runs the Saluki model to predict transcript half-life from sequence and splicing info
## See 261022 for details of how the following code was developed

#    Copyright 2026 eugenekwaNeuromics

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

r"""Infer transcript half-life using the Saluki model. 

Requires that the target RNA sequence(s) be encoded into a one-hot (6, 12288) tensor:

    * Rows 1-4: ACGU identity.
    * Row 5: First base per codon (5' -> 3').
    * Row 6: Last base per exon (5' -> 3').

All '-' strand sequences are converted to their reverse complement to make obtaining rows 5-6 easier.

We obtain this information from relevant Gene Transfer Format file(s) / GTF(s) and paired genome FASTA. 
Will assume there exists a GTF with the standard CDS, exon, genomic coordinate information etc (the "annotation GTF";
and another GTF with exon coordinates for novel transcripts (the "observed GTF"). 
If ignoring novel transcript evaluation, these two GTFs are interchangeable.

Returns half-life predictions per transcript as unit-less "relative half-life", specifically PC1 of standardized (z-scored) log half-lives obtained from training data.
As there are 50 models (trained on different folds and crosses), there should be 50 predictions. 

Please cite the original authors' work.

> Agarwal, Vikram, and David R. Kelley. "The genetic and biochemical determinants of mRNA degradation rates in mammals." Genome biology 23.1 (2022): 245.

Examples
--------
    python run_saluki.py -o out_dir/ -b basenji/ -w saluki_weights/train_gru/ -i extended_annotations.gtf -a genome.gtf -f genome.fa -g ENSMUSGXXX -g ENSMUSGXXX

Notes
-----
Dependencies:

    * tensorflow
    * numpy
    * polars
    * pysam

Tested with python 3.12.

Seed is fixed at 42 for reproducibility.

Currently only tested on small gene subsets; unclear about performance on whole transcriptome at once.

For details about required inputs:

    python run_saluki.py --help

Licenses
--------
This script uses code and resources from the following projects:

    * Basenji (Apache License 2.0): https://github.com/calico/basenji/blob/master/LICENSE 
    * Saluki model weights (Creative Commons Attribution 4.0 International): https://zenodo.org/records/6326409

Please refer to the respective licences for terms of use.
"""

## Libraries ----------------------------------------------------------------------------------------
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
try:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
except Exception as e:
    raise ImportError(f'Unable to import tensorflow -- make sure you are in the correct env: {e}')
import numpy as np
import polars as pl
import sys
import json
import glob
import gzip
import pathlib
import pysam
import argparse


## Functions ----------------------------------------------------------------------------------------
def build_one_hots(
    gtf_observed: str, 
    gtf_annot: str,
    fa_genome: str,
    n_bases: int = 12288,
    selected_gene_ids: list[str] | None = None,
) -> dict:
    r"""Get all the info we will need for Saluki.

    Parameters
    ----------
    gtf_observed: str
        The GTF file containing all observed transcripts (including novel ones).
    gtf_annot: str
        The original GTF file (e.g. ENSEMBL) that provides transcript aliases and CDS information.
    fa_genome: str
        The FASTA file matched with the chosen GTF(s). Ensure this has been gunzipped (or was gzip with `bgzip`).
    n_bases: int
        The number of bases that the model needs.
    selected_gene_ids: list[str], optional
        For speed, would be good to specify which gene to focus on. 
        
    Returns
    -------
    onehot: np.ndarray
        One hot tensor of dimensions (`n_bases`, 6): where first four channels indicate ACGT, 
        fifth indicates the first base per codon, and sixth indicates the last base per exon. 
    metadata: pl.DataFrame
        Metadata per transcript.
    """
    if selected_gene_ids is not None:
        if not isinstance(selected_gene_ids, list):
            selected_gene_ids = [selected_gene_ids]
        if len(selected_gene_ids) == 0:
            selected_gene_ids = None

    ## We will need the fasta file to get sequence
    if not pathlib.Path(f'{fa_genome}.fai').is_file():
        try:
            pysam.faidx(fa_genome)
        except Exception as e:
            raise ValueError(f'Unable to index {fa_genome}: {e}')
    fa = pysam.FastaFile(fa_genome)

    ## Extract information about which exons are used per transcript
    ## Keep everything as LazyFrame as much as possible to keep memory usage low
    gtf = pl.scan_csv(
        gtf_observed, comment_prefix='#', has_header=False, separator='\t', 
        new_columns=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'],
        schema_overrides={'seqname': pl.String, 'start': pl.Int64, 'end': pl.Int64}
    )
    gtf = gtf.with_columns(pl.col('attribute').str.extract(r'gene_id\s+"([^"]+)"', 1).alias('gene_id'))
    gtf = gtf.with_columns(pl.col('gene_id').str.replace(r'\.\d+$', '').alias('gene_id_cleaned'))
    if selected_gene_ids is not None:  
        gtf = gtf.filter(pl.col('gene_id').is_in(selected_gene_ids) | pl.col('gene_id_cleaned').is_in(selected_gene_ids))
    exons = gtf.filter(pl.col('feature')=='exon')
    exons = exons.with_columns(pl.col('attribute').str.extract(r'transcript_id\s+"([^"]+)"', 1).alias('transcript_id'))
    exons = exons.with_columns(pl.col('transcript_id').str.replace(r'\.\d+$', '').alias('transcript_id_cleaned'))
    exons = exons.with_columns(pl.col('attribute').str.extract(r'exon_number\s+"?(\d+)"?', 1).cast(pl.Int64).alias('exon_number'))

    ## Get sequence per exon
    def _get_region_seqs(gcoord: list[pl.Series]) -> pl.Series:
        r"""Fetches sequences per exon."""
        chroms = gcoord[0]
        starts = gcoord[1]
        ends = gcoord[2]
        strands = gcoord[3]
        seq = [fa.fetch(chrom, start-1, end).lower() for chrom, start, end in zip(chroms, starts, ends)]
        seq = [pysam.reverse_complement(seq[i]) if strands[i] == '-' else seq[i] for i in range(len(seq))]
        seq = [s[:-1] + s[-1].upper() for s in seq]
        return pl.Series(name='SEQ', values=seq, dtype=pl.String) 
    exons = exons.with_columns(
        pl.map_batches(
            exprs=['seqname', 'start', 'end', 'strand'], 
            function=_get_region_seqs, 
            return_dtype=pl.String
        ).alias('fetched_seq')
    )

    ## Obtain the sequence per transcript
    def _one_hot_early(seq: str) -> np.ndarray:
        r"""Start constructing a 6 channel one-hot per sequence before CDS information is known."""
        encoded = np.zeros((n_bases, 6), dtype=np.uint8)
        base_map = {'A':0, 'C':1, 'G':2, 'T':3, 'U':3} 
        for i, base in enumerate(seq):
            if i >= encoded.shape[0]:
                continue
            if base.isupper():
                encoded[i,5] = 1
            channel = base_map.get(base.upper(), None)
            if channel is not None:
                encoded[i, channel] = 1
        return encoded
    transcript_seqs = (
        exons.sort(['gene_id_cleaned', 'transcript_id_cleaned', 'exon_number'])
        .group_by(['gene_id', 'gene_id_cleaned', 'transcript_id', 'transcript_id_cleaned'], maintain_order=True)
        .agg(pl.col('fetched_seq').str.join('').alias('splice_seq'))
        .with_columns(
            pl.col('splice_seq')
            .map_elements(function=_one_hot_early, return_dtype=pl.Object)
            .alias('onehot')
        )
    )
    
    ## Go back to the original GTF and get the CDS and as before, and get the sequence per CDS region
    gtfa = pl.scan_csv(
        gtf_annot, comment_prefix='#', has_header=False, separator='\t', 
        new_columns=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'],
        schema_overrides={'seqname': pl.String, 'start': pl.Int64, 'end': pl.Int64}
    )
    gtfa = gtfa.with_columns(pl.col('attribute').str.extract(r'gene_id\s+"([^"]+)"', 1).alias('gene_id'))
    gtfa = gtfa.with_columns(pl.col('gene_id').str.replace(r'\.\d+$', '').alias('gene_id_cleaned'))
    if selected_gene_ids is not None:  
        gtfa = gtfa.filter(pl.col('gene_id').is_in(selected_gene_ids) | pl.col('gene_id_cleaned').is_in(selected_gene_ids))
    cds = gtfa.filter(pl.col('feature')=='CDS')
    cds = cds.with_columns(pl.col('attribute').str.extract(r'transcript_id\s+"([^"]+)"', 1).alias('transcript_id'))
    cds = cds.with_columns(pl.col('transcript_id').str.replace(r'\.\d+$', '').alias('transcript_id_cleaned'))
    cds = cds.with_columns(pl.col('attribute').str.extract(r'exon_number\s+"?(\d+)"?', 1).cast(pl.Int64).alias('exon_number'))
    cds = (
        cds.with_columns(
            pl.map_batches(
                exprs=['seqname', 'start', 'end', 'strand'], 
                function=_get_region_seqs, 
                return_dtype=pl.String
            ).alias('fetched_seq')
        )
        .sort(['gene_id_cleaned', 'transcript_id_cleaned', 'exon_number'])
        .group_by(['gene_id', 'gene_id_cleaned', 'transcript_id', 'transcript_id_cleaned'], maintain_order=True)
        .agg(pl.col('fetched_seq').str.join('').alias('cds_seq'))
    )

    ## Combine above
    cds_by_gene = cds.group_by('gene_id_cleaned').agg(pl.col('cds_seq').unique()).collect().rows_by_key(key='gene_id_cleaned', unique=True)
    transcript_seqs = transcript_seqs.join(cds.select('cds_seq','transcript_id_cleaned'), on='transcript_id_cleaned', how='left')

    ## Also get the transcript names    
    tx_info = gtfa.filter(pl.col('feature')=='transcript')
    tx_info = tx_info.with_columns(pl.col('attribute').str.extract(r'transcript_id\s+"([^"]+)"', 1).alias('transcript_id'))
    tx_info = tx_info.with_columns(pl.col('attribute').str.extract(r'transcript_name\s+"([^"]+)"', 1).alias('transcript_name'))
    tx_info = tx_info.with_columns(pl.col('transcript_id').str.replace(r'\.\d+$', '').alias('transcript_id_cleaned'))
    tx_info = tx_info.unique(subset=['transcript_id_cleaned'], keep='first')
    # tx_info = tx_info.collect()
    transcript_seqs = transcript_seqs.join(
        tx_info.select(['transcript_id_cleaned', 'transcript_name']),
        on='transcript_id_cleaned', how='left'
    ).sort('transcript_name', nulls_last=False)

    ## Pull out the onehots since polars does not know how to include them in a struct
    transcript_seqs = transcript_seqs.collect()
    onehots = transcript_seqs['onehot'].to_list()
    transcript_seqs = transcript_seqs.drop('onehot')
    
    ## Do a quick exact alignment
    def _pairwise_align(info: list[pl.Series]):
        r"""Performs pairwise alignment requiring exact match (use rapidfuzz if want imprecise match to be tolerated)."""
        gene_id = info['gene_id_cleaned']
        exon_seq = str(info['splice_seq']).upper()
        cds_seq = info['cds_seq']
        onehot = info['onehot']
        if cds_seq is None:
            candidates = cds_by_gene.get(gene_id, [])
            if len(candidates) == 0:
                return onehot
            if isinstance(candidates[0], list):
                candidates = [item for sublist in candidates for item in sublist]
            matches = [seq.upper() for seq in candidates if seq and seq.upper() in exon_seq]
            if len(matches)==0:
                return onehot
            cds_seq = max(matches, key=len)
        cds_seq = str(cds_seq).upper()
        start = exon_seq.find(cds_seq)
        if start == -1:
            return onehot
        codon_starts = np.arange(start, min(onehot.shape[0], len(cds_seq) + start + 3), 3)
        onehot[codon_starts, 4] = 1
        return onehot

    ## Loop in collected info to get info we need
    final_onehots = []
    for row, onehotx in zip(transcript_seqs.iter_rows(named=True), onehots):
        final_onehots.append(
            _pairwise_align({
                'gene_id_cleaned': row['gene_id_cleaned'],
                'splice_seq': row['splice_seq'],
                'cds_seq': row['cds_seq'],
                'onehot': onehotx,
            })
        )

    ## Keep the minimum needed info
    transcript_seqs_final = transcript_seqs.select(['gene_id', 'gene_id_cleaned', 'transcript_id', 'transcript_id_cleaned', 'transcript_name'])
    return {
        'metadata': transcript_seqs_final,
        'onehots': np.stack(final_onehots),
    }


## MAIN ---------------------------------------------------------------------------------------------
def main():
    ## Parse arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='Output directory to store results.')
    parser.add_argument('-b', '--basenji_dir', required=True, type=str, help='Location of basenji directory (obtained e.g. by git clone).')
    parser.add_argument('-w', '--saluki_weights_dir', required=True, type=str, help='Directory containing Saluki model weights.')
    parser.add_argument('-i', '--input_gtf', required=True, type=str, help='GTF file containing observed transcripts (including novel ones)')
    parser.add_argument('-a', '--annotation_gtf', required=True, type=str, help='GTF file containing CDS information. If same as input GTF file, then specify the input GTF file here too.')
    parser.add_argument('-f', '--genome_fasta', required=True, type=str, help='Genome FASTA file for obtaining sequences. Ensure this is unzipped.')
    parser.add_argument('-s', '--species', type=str, help='Choice of model species: "human", "mouse", or "both" (default).', default='both')
    parser.add_argument('-p', '--file_prefix', type=str, help='Prefix of out file.', default="")
    parser.add_argument('-g', '--gene', type=str, action='append', help='Flag for specifying which genes to focus on. Alternatively a single .csv file with a "gene" column.')
    args = parser.parse_args()

    ## Param checks
    if not pathlib.Path(args.basenji_dir).is_dir():
        raise FileNotFoundError(f'{args.basenji_dir} directory not found')
    if not pathlib.Path(args.saluki_weights_dir).is_dir():
        raise FileNotFoundError(f'{args.saluki_weights_dir} directory not found')
    if not pathlib.Path(args.input_gtf).is_file():
        raise FileNotFoundError(f'{args.input_gtf} file not found')
    if not pathlib.Path(args.annotation_gtf).is_file():
        raise FileNotFoundError(f'{args.annotation_gtf} file not found')
    if not pathlib.Path(args.genome_fasta).is_file():
        raise FileNotFoundError(f'{args.genome_fasta} file not found')
    valid_species = {
        'human': 'human',
        'h': 'human',
        'hs': 'human',
        '0': 'human',
        
        'mouse': 'mouse',
        'm': 'mouse',
        'mm': 'mouse',
        '1': 'mouse',
        
        'both': 'both',
        'b': 'both',
    }
    species = valid_species.get(args.species.lower(), None)
    if species is None:
        raise ValueError(f'Invalid species provided: {args.species.lower()}')
        
    ## Create output dir if not already there
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ## Load the saluki model
    sys.path.append(args.basenji_dir)
    try:
        from basenji import rnann
    except Exception as e:
        raise ImportError(f'Could not import the RNANN module from basenji: {e}')
    params_file = f'{args.basenji_dir}/manuscripts/saluki/params.json'
    if not pathlib.Path(params_file).is_file():
        raise FileNotFoundError(f'Could not find the params.json for saluki model: looked for {params_file}')
    with open(params_file) as params_open:
        params = json.load(params_open)
    if not isinstance(params['model']['num_targets'], list):
        params['model']['num_targets'] = [params['model']['num_targets']] * params['model'].get('heads', 1)

    ## Parse the gene info
    gene_info = args.gene
    if isinstance(gene_info, list):
        if len(gene_info) == 1:
            gene_info = gene_info[0]
    if isinstance(gene_info, str):
        if '.csv' in gene_info and pathlib.Path(gene_info).is_file():
            gidf = pl.scan_csv(gene_info)
            if 'gene' not in gidf.collect_schema().keys():
                raise TypeError(f'"gene" column not detected in the gene file.')
            gene_info = gidf.select('gene').collect().to_numpy().ravel().tolist()
    if not isinstance(gene_info, list):
        if not isinstance(gene_info, str):
            raise TypeError(f'gene_info is neither list nor str: it is {type(gene_info)}')
        gene_info = [gene_info]
        
    ## Get the inputs for saluki
    res = build_one_hots(
        gtf_observed = args.input_gtf,
        gtf_annot = args.annotation_gtf,
        fa_genome = args.genome_fasta,
        selected_gene_ids = gene_info,
        n_bases=int(params['model']['seq_length']),
    )
    rna_1hot = res['onehots']
    rna_meta = res['metadata']

    ## Initialize model
    saluki = rnann.RnaNN(params['model'])

    ## Get predictions
    for default_species in ['human', 'mouse']:
        sp = species
        if sp == 'both':
            sp = default_species
        elif sp != default_species:
            continue
        else:
            sp = default_species
            
        if sp == 'mouse':
            head_i = 1
        elif sp == 'human':
            head_i = 0
        else:
            raise ValueError(f'Only mouse or human heads available')
        
        model_str = f'model{head_i}_best.h5'
        predictions = []
        num_folds = len(glob.glob('%s/f*_c0/train/%s' % (args.saluki_weights_dir, model_str)))
        num_crosses = len(glob.glob('%s/f0_c*/train/%s' % (args.saluki_weights_dir, model_str)))
        # print('Folds %d, Crosses %d' % (num_folds, num_crosses))
        for fi in range(num_folds):
            for ci in range(num_crosses):
              model_file = '%s/f%d_c%d/train/%s' % (args.saluki_weights_dir, fi, ci, model_str)
              saluki.restore(model_file) 
              saluki.build_ensemble(ensemble_shifts=[0]) 
              preds_fc = saluki.predict(rna_1hot)
              predictions.append(preds_fc)
        predictions = np.array(predictions)
        predictions = np.transpose(predictions, [1,2,0])
        out_file = f'{args.output_dir}/{args.file_prefix}preds_Saluki_{sp}.csv.gz'
        pl.concat(
            (rna_meta, pl.DataFrame((predictions).squeeze(1))), how='horizontal'
        ).write_csv(out_file, compression='gzip')

## RUN ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
  main()