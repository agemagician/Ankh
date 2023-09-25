import os
import argparse
from Bio import SeqIO
import pandas as pd
import numpy as np


def sav_stats(gen_id,gen_aln,vespa_map_out):
    sav_vector = []
    c = 0
    df = pd.read_csv(vespa_map_out)
    seq = gen_aln

    for j in seq:
        if j == '_':
            sav_vector.append(0)
        else:
            c=c+1
            if c-1 in list(df['Gen Seq Index'].values):
                sav_vector.append(list(df['Score'].values)[list(df['Gen Seq Index'].values).index(c-1)]*10)
            else:
                sav_vector.append(0)
    return sav_vector

def cons_stats(ref_id,ref_aln,cons_class):
    cons_class = cons_class.split(',')
    ref_seq = ref_aln
    cons_vector = []
    c = 0
    for j in ref_seq:
        if j == '_':
            cons_vector.append(0)
        else:
            c=c+1
            cons_vector.append(int(cons_class[c-1]))
    return cons_vector
def arg_parser():
    parser = argparse.ArgumentParser(description='Computes SAV and Conservation scores statistics')
    parser.add_argument('--vespa_map_out', type=str, action='store', help='FULL Path of vespa mapped score output csv file',required=True)
    parser.add_argument('--ref_id', type=str, action='store', help='reference sequence ID',required=True)
    parser.add_argument('--ref_aln', type=str, action='store', help='MSA reference sequence',required=True)
    parser.add_argument('--cons_class', type=str, action='store', help='Conservation classes of reference sequence',required=True)
    parser.add_argument('--gen_id', type=str, action='store', help='generated sequence ID',required=True)
    parser.add_argument('--gen_aln', type=str, action='store', help='MSA generated sequence',required=True)
    args = parser.parse_args()
    
    mapped_scores = args.vespa_map_out
    ref_id = args.ref_id
    ref_aln = args.ref_aln
    cons_class = args.cons_class
    gen_id = args.gen_id
    gen_aln = args.gen_aln
    return [mapped_scores,ref_id,ref_aln,cons_class,gen_id,gen_aln]

##parsing inputs:
inputs = arg_parser()
mapped_scores = inputs[0]
ref_id = inputs[1]
ref_aln = inputs[2]
cons_class = inputs[3]
gen_id = inputs[4]
gen_aln = inputs[5]

sav_vector = sav_stats(gen_id,gen_aln,mapped_scores)
cons_vector = cons_stats(ref_id,ref_aln,cons_class)

os.system('mkdir -p vespa_msa')
os.chdir('vespa_msa')
os.system('mkdir -p sav_stats')
os.system('mkdir -p cons_stats')
sav_score_path = os.path.join('sav_stats',gen_id+'_sav_scores.csv')
cons_score_path = os.path.join('cons_stats',ref_id+'_cons_scores.csv')

with open (sav_score_path, 'w') as s:
    s.write(gen_id+','+str(sav_vector)[1:-1]+'\n')
with open (cons_score_path, 'w') as s:
    s.write(ref_id+','+str(cons_vector)[1:-1]+'\n')
