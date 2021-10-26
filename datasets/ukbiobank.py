from os.path import join, exists
import os
import pandas as pd
import numpy as np
from pandas_plink import read_plink1_bin
# TODO
def ukbiobank(path, save=True):
    """
    Download, parse and process UKBiobank data
    Examples
    --------
    from ccagame import datasets

    train_view_1, train_view_2 = datasets.ukbiobank()

    Returns
    -------
    train_view_1, train_view_2
    """
    if exists(join(path, 'genetics_processed.npy')) and exists(join(path, 'brain_processed.npy')):
        brain_train = np.load('brain_processed.npy')
        genetics_train = np.load('genetics_processed.npy')
    else:
        #load brain data with effect of ICV removed using HBR 
        load_data = False
        if load_data:
            print(os.getcwd())
            brain_path = '/mnt/c/Users/anala/Documents/PhD/year_1/project_work/UK_Bio/data/210511'
            genetics_path = '/mnt/c/Users/anala/Documents/PhD/year_1/project_work/UK_Bio/data/210511'
            brain_train =  pd.read_csv(join(brain_path, 'brain_data.csv'), header=0, index_col=0)
            #find caucasian subj
            caucasian_subj = pd.read_csv(join(genetics_path, 'ukb_caucasian.txt'), sep=' ')
            brain_train = brain_train[brain_train.index.isin(caucasian_subj.iloc[:,0])]
            brain_subj = brain_train.index
            brain_train = brain_train.to_numpy()

            #load SNP data filtered with maf>1% and missingness<2%

            #SNP_files = {'bed': 'ukb_cal_merged_maf01_geno02.bed', 
            #'bim': 'ukb_cal_merged_maf01_geno02.bim', 
            #'fam': 'ukb_cal_merged_maf01_geno02.fam',
            #'freq':'ukb_cal_merged_maf01_geno02.freq' }
            SNP_files = {'bed': 'ukb_cal_merged_maf0.01_geno0.2_sub_snps_prune0.1.bed', 
            'bim': 'ukb_cal_merged_maf0.01_geno0.2_sub_snps_prune0.1.bim', 
            'fam': 'ukb_cal_merged_maf0.01_geno0.2_sub_snps_prune0.1.fam',
            'frq': 'ukb_cal_merged_maf0.01_geno0.2_sub_snps_prune0.1.csv' }
            
            SNP_bed = join(genetics_path, SNP_files['bed'])
            SNP_bim = join(genetics_path, SNP_files['bim'])
            SNP_fam = join(genetics_path, SNP_files['fam'])     
            MAF_data = pd.read_csv(join(genetics_path, SNP_files['frq']), header=0)
            MAF = MAF_data['MAF'].to_numpy().transpose()
            gen_data = read_plink1_bin(SNP_bed, SNP_bim, SNP_fam, verbose=False)
            genetics_train = np.array(gen_data.values)

            SNP_subj = pd.DataFrame(gen_data['sample'].coords['sample'].values, columns=['IID'])
            SNP_subj['IID'] = SNP_subj['IID'].astype(int)
            SNP_subj = SNP_subj[SNP_subj['IID'].isin(brain_subj)]
            genetics_train = genetics_train[SNP_subj.index,:]

            #fill empty entries with MAF
            #MAF.reshape((1,MAF.size))
            genetics_train = MAF*(np.isnan(genetics_train).astype(int)) + np.nan_to_num(genetics_train)
            if save:
                np.save(join(path, 'brain_train.npy'), brain_train)
                np.save(join(path, 'genetics_train.npy'), genetics_train)

    brain_train = pd.read_csv(join(path, 'Epilepsy_MRI_train_zscores.csv'), header=None).to_numpy()
    genetics_train = pd.read_csv(join(path, 'Epilepsy_genetics_train_processed.csv'), header=None).to_numpy()
    
    return brain_train, genetics_train
