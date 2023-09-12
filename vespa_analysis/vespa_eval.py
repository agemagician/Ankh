import os
import argparse
from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import numpy as np

def fasta2dict(fasta_file):
    fasta_parsed = SeqIO.parse(fasta_file,'fasta')
    fasta_dict = dict()
    for record in fasta_parsed:
        fasta_dict[record.id] = str(record.seq)
    return fasta_dict

def run_vespa(fasta_file,cache_path):
    cwd = os.getcwd()
    os.system('vespa '+fasta_file+' --prott5_weights_cache '+cache_path+' --output_classes')
    vespa_out = cwd+'/vespa_run_directory/output'
    return vespa_out

def run_mmseq(gen_fasta,ref_fasta,gen_name,threads):
    cwd = os.getcwd()
    os.system('mkdir -p '+gen_name)
    os.chdir(gen_name)
    os.system('mmseqs createdb '+gen_fasta+' '+gen_name+'_db')
    os.system('mmseqs createdb '+ref_fasta+' '+ref_name+'_db')
    os.system('mmseqs search '+gen_name+'_db '+ref_name+'_db res tmp --threads '+str(threads))
    os.system('mmseqs align '+gen_name+'_db '+ref_name+'_db res '+gen_name+'_aln -a')
    os.system('mmseqs convertalis '+gen_name+'_db '+ref_name+'_db '+gen_name+'_aln '+gen_name+'_aln.tab --format-output query,target,qaln,taln')
    os.system('cp '+gen_name+'_aln.tab '+cwd)
    os.chdir(cwd)
    os.system('rm -r '+gen_name)
    mmseq_out = cwd+'/'+gen_name+'_aln.tab'
    return mmseq_out

def generate_msa(ref_fasta,gen_fasta,threads):
    os.system('mkdir -p vespa_msa')
    os.system('cat '+ref_fasta+' '+gen_fasta+' > ref_gen.fasta')
    os.system('clustalo -i ref_gen.fasta --threads='+str(threads)+' --iter=2 --full-iter -o vespa_msa/ref_gen_msa.fasta -v --force')
    ref_gen_msa = 'vespa_msa/ref_gen_msa.fasta'
    return ref_gen_msa

## Function for Pairwise Mapping VESPA scores to Mutants ##
def pair_map (gen_dict,ref_dict,mmseq_out,vespa_out,threads):
    ref_id = ''
    ref_seq = ''
    gen_id = ''
    gen_seq = ''

    with open ('vespa_pair_cmd.list','w') as cmd:
        for i in range(len(gen_dict)):
            ref_id = list(ref_dict.keys())[i]
            ref_seq = ref_dict[ref_id]
            gen_id = list(gen_dict.keys())[i]
            gen_seq = gen_dict[gen_id]
            cmd.write('python vespa_pair.py --mmseq_out '+mmseq_out+' --vespa_out '+vespa_out+'/'+str(i)+'.csv --ref_id '+ref_id+' --ref_seq '+ref_seq+' --gen_id '+gen_id+' --gen_seq '+gen_seq+'\n')

    os.system('parallel -j '+str(threads)+' < vespa_pair_cmd.list')

## Function for MSA Mapping VESPA scores and Conservation Classes ##
def msa_map (ref_fasta,gen_fasta,threads):
    ref_gen_msa = generate_msa(ref_fasta,gen_fasta,threads)
    msa_dict = fasta2dict(ref_gen_msa)
    cons_fasta = cwd+'/vespa_run_directory/conspred_class.fasta'
    cons_dict = fasta2dict(cons_fasta)
    mapped_scores = cwd+'/vespa_scores'

    with open ('vespa_msa_cmd.list','w') as cmd:
        for i in range(len(gen_dict)):
            ref_id = list(ref_dict.keys())[i]
            gen_id = list(gen_dict.keys())[i]
            ref_aln = msa_dict[ref_id]
            gen_aln = msa_dict[gen_id]
            cons_class = cons_dict[ref_id]
            vespa_map_out = cwd+'/vespa_scores/'+gen_id+'_vespa_scores.csv'
            cmd.write('python vespa_msa.py --vespa_map_out '+vespa_map_out+' --ref_id '+ref_id+' --ref_aln '+ref_aln.replace('-','_')+' --cons_class '+cons_class+' --gen_id '+gen_id+' --gen_aln "'+gen_aln.replace('-','_')+'"\n')
    os.system('parallel -j '+str(threads)+' < vespa_msa_cmd.list')   
    os.system('cat vespa_msa/cons_stats/*.csv > vespa_msa/'+gen_name+'_cons_scores.csv')
    os.system('rm -r vespa_msa/cons_stats/')
    os.system('cat vespa_msa/sav_stats/*.csv > vespa_msa/'+gen_name+'_sav_scores.csv')
    os.system('rm -r vespa_msa/sav_stats/')

parser = argparse.ArgumentParser(description='Calculates SAV scores using VESPA')
parser.add_argument('-r', type=str, action='store', help='FULL Path Fasta file containing the Reference sequences',required=True)
parser.add_argument('-g', type=str, action='store', help='FULL Path Fasta file containing the Generated sequences',required=True)
parser.add_argument('--cache', type=str, action='store', help='FULL Path of caching folder to store Prott5 weights',required=True)
parser.add_argument('--threads', type=str, action='store', help='Number of threads to be used in parallelization',required=True)
args = parser.parse_args()

## Parsing Input ##
ref_fasta = args.r
ref_name = ref_fasta.split('/')[-1]
ref_name = ref_name.split('.')[0]
gen_fasta = args.g 
gen_name = gen_fasta.split('/')[-1]
gen_name = gen_name.split('.')[0]
cache_path = args.cache
threads = int(args.threads)
cwd = os.getcwd()

ref_dict = fasta2dict(ref_fasta)
gen_dict = fasta2dict(gen_fasta)

dummy_thread = 0
if len(gen_dict)%threads != 0:
    dummy_thread = threads
    for i in range(threads):
        dummy_thread = dummy_thread-1
        if len(gen_dict)%dummy_thread ==0:
            threads = dummy_thread
            break

## Running VESPA and MMSEQ ##
vespa_out = run_vespa(ref_fasta,cache_path)
mmseq_out = run_mmseq(gen_fasta,ref_fasta,gen_name,threads)

## Mapping VESPA scores to Mutants ##
pair_map (gen_dict,ref_dict,mmseq_out,vespa_out,threads)

## Computing SAV and Conservation statistics ##
msa_map (ref_fasta,gen_fasta,threads)

## Plotting ##
df_sav = pd.read_csv('vespa_msa/'+gen_name+'_sav_scores.csv',header=None)
df_sav_avg = df_sav.mean()
sav_avg_arr = np.array(list(df_sav_avg.values))
df_cons = pd.read_csv('vespa_msa/'+gen_name+'_cons_scores.csv',header=None)
df_cons_avg = df_cons.mean()
cons_avg_arr = np.array(list(df_cons_avg.values))
spearman_corr = scipy.stats.mstats.spearmanr(cons_avg_arr,sav_avg_arr)

df_all_avg = pd.DataFrame()
df_all_avg['Average Mutant SAV Scores'] = list(df_sav_avg.values)
df_all_avg['Average Wildtype Conservation Class'] = list(df_cons_avg.values)

## Plotting Positional SAV Effect and Conservation Classes ##
plt.figure(figsize=(15,8))
sns.set(style="darkgrid")
sns.lineplot(data=df_all_avg)
plt.xlabel('MSA Position Index', fontsize=16)
plt.title('Positional SAV Effect and Conservation')
plt.legend() 
plt.savefig(gen_name+'_sav_cons_score.png')

## Plotting Relation Between SAV Effect Scores and Conservation Class Per Position ##
plt.figure(figsize=(15,8))
sns.set(style="whitegrid")
p = sns.regplot(x= list(df_all_avg['Average Mutant SAV Scores'].values),y=list(df_all_avg['Average Wildtype Conservation Class'].values),color="lightseagreen")
#calculate slope and intercept of regression equation
slope, intercept, r, p, sterr = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),y=p.get_lines()[0].get_ydata())
#add regression equation to plot
plt.text(0.25, 7, 'y = ' + str(round(intercept,3)) + ' + (' + str(round(slope,3)) + ')x', fontsize=15, color='teal',fontstyle = "oblique")
## Spearman Correlation Between  Postional Average SAV scores and Average Conservation Classes ##
plt.text(0.25, 6, 'Spearman correlation = '+str(round(spearman_corr.statistic,3)), fontsize=15, color='teal',fontstyle = "oblique")
plt.xlabel('Positional SAV Scores', fontsize=14)
plt.ylabel('Positional Conservation Scores', fontsize=14)
plt.savefig(gen_name+'_corr_score.png')