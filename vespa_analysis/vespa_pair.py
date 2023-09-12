import os
import argparse
from Bio import SeqIO
import pandas as pd

def gen_score(ref_seq,gen_seq,ref_aln,gen_aln,vesp_dict):
    scores = dict()
    c = 0
    gc = 0
    g_inx =[]
    g_start = 0
    if gen_aln == ref_aln:
        scores['NA'] = 0
        g_inx.append(0)
    else:
        if len(gen_seq) != len(gen_aln.replace('-','')):
            g_inx_ter = gen_seq.replace(gen_aln.replace('-',''),'-').split('-')
            g_start = len(g_inx_ter[0])
        if len(ref_seq) != len(ref_aln.replace('-','')):
            ref_ter_list = ref_seq.replace(ref_aln.replace('-',''),'-').split('-')
            gen_ter_list = ['-'*len(ref_ter_list[0]), '-'*len(ref_ter_list[1])]
            ref_aln = ref_ter_list[0]+ref_aln+ref_ter_list[1]
            gen_aln = gen_ter_list[0]+gen_aln+gen_ter_list[1]

        for j in range(len(ref_aln)):
            if gen_aln[j]!='-':
                gc = gc+1
            if ref_aln[j] != '-':
                c = c+1
                if (gen_aln[j] != '-') and (ref_aln[j] != gen_aln[j]):
                    scores[ref_aln[j]+str(c-1)+gen_aln[j]] = vesp_dict[ref_aln[j]+str(c-1)+gen_aln[j]]
                    g_inx.append(gc-1+g_start)
    return [scores,ref_aln,gen_aln,g_inx]

parser = argparse.ArgumentParser(description='Maps VESPA scores to mutant sequences')
parser.add_argument('--mmseq_out', type=str, action='store', help='FULL Path of mmseqs alignment output',required=True)
parser.add_argument('--vespa_out', type=str, action='store', help='FULL Path of vespa output csv file',required=True)
parser.add_argument('--ref_id', type=str, action='store', help='reference sequence ID',required=True)
parser.add_argument('--ref_seq', type=str, action='store', help='reference sequence',required=True)
parser.add_argument('--gen_id', type=str, action='store', help='generated sequence ID',required=True)
parser.add_argument('--gen_seq', type=str, action='store', help='generated sequence',required=True)
args = parser.parse_args()

mmseq_out = args.mmseq_out
vesp_out = args.vespa_out
ref_id = args.ref_id
ref_seq = args.ref_seq
gen_id = args.gen_id
gen_seq = args.gen_seq

df_mm = pd.read_csv(mmseq_out,sep='\t',header=None)
df_dummy = df_mm.loc[(df_mm[0]==gen_id) & (df_mm[1]==ref_id)]
gen_aln = str(df_dummy.iat[0,2])
ref_aln = str(df_dummy.iat[0,3])

df_vesp = pd.read_csv(vesp_out,sep=';')
vesp_dict=dict()
vesp_dict = {list(df_vesp['Mutant'].values)[i]: list(df_vesp['VESPAl'].values)[i] for i in range(len(list(df_vesp['Mutant'].values)))}

out_list = gen_score(ref_seq,gen_seq,ref_aln,gen_aln,vesp_dict)
scores = out_list[0]
g_inx = out_list[3]
os.system('mkdir -p vespa_scores')
df_out = pd.DataFrame()
df_out['Mutation'] = list(scores.keys())
df_out['Score'] = list(scores.values())
df_out['Gen Seq Index'] = g_inx
df_out.to_csv('vespa_scores/'+gen_id+'_vespa_scores.csv',index=False)

with open ('vespa_scores/'+gen_id+'_alignment.fasta','w') as aln:
    aln.write('>'+ref_id+'\n')
    aln.write(out_list[1]+'\n')
    aln.write('>'+gen_id+'\n')
    aln.write(out_list[2]+'\n')