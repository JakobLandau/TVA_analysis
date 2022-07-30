###############################################################################
#                               IMPORTS                                       #
###############################################################################
import os
import glob
import time
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.multitest

###############################################################################
#                               CONSTANTS                                     #
###############################################################################
RAW_DATA_PATH = "raw_data"
OUTPUT_DIR_PATH = "output"
COSMIC_ROLE_IN_CANCER_DIC = {'fusion': ['ACSL3', 'ACSL6', 'AFF1', 'AKAP9', 'ALDH2', 'ASPSCR1', 'ATIC', 'BCL7A', 'BCR', 'C15orf65', 'CANT1', 'CEP89', 'CHCHD7', 'CHIC2', 'CLIP1', 'CLP1', 'CNTRL', 'COL1A1', 'COL2A1', 'COL3A1', 'COX6C', 'CRTC3', 'DCTN1', 'DNAJB1', 'DUX4L1', 'EIF4A2', 'ELN', 'EML4', 'ERC1', 'EZR', 'FAM131B', 'FGFR1OP', 'FIP1L1', 'FNBP1', 'GAS7', 'GMPS', 'GOLGA5', 'GOPC', 'GPHN', 'HERPUD1', 'HIST1H4I', 'HLA-A', 'HMGN2P46', 'HOOK3', 'HSP90AA1', 'HSP90AB1', 'IGH', 'IGK', 'IGL', 'IL2', 'IL21R', 'ITK', 'JAZF1', 'KDSR', 'KIAA1549', 'KIF5B', 'KLK2', 'KTN1', 'LASP1', 'LCP1', 'LHFPL6', 'LIFR', 'LMNA', 'LSM14A', 'MDS2', 'MLLT1', 'MLLT11', 'MLLT3', 'MLLT6', 'MNX1', 'MSN', 'MUC1', 'MYH11', 'MYO5A', 'NACA', 'NCKIPSD', 'NCOA1', 'NFIB', 'NIN', 'NONO', 'NSD1', 'NUMA1', 'NUP214', 'NUTM2B', 'NUTM2D', 'OMD', 'PAFAH1B2', 'PAX7', 'PAX8', 'PCM1', 'PDE4DIP', 'PICALM', 'PPFIBP1', 'PRCC', 'PRRX1', 'PWWP2A', 'RABEP1', 'RALGDS', 'RBM15', 'RNF213', 'RPN1', 'S100A7', 'SDC4', 'SEPT5', 'SEPT6', 'SEPT9', 'SHTN1', 'SLC45A3', 'SNX29', 'SPECC1', 'SRGAP3', 'SS18', 'SS18L1', 'STRN', 'TCEA1', 'TCF12', 'TFG', 'TFPT', 'TFRC', 'THRAP3', 'TMPRSS2', 'TOP1', 'TPM4', 'TPR', 'TRA', 'TRB', 'TRD', 'TRIP11', 'VAV1', 'VTI1A', 'WDCP', 'ZCCHC8', 'ZMYM2', 'ZNF384'],
                             'tsg': ['ABI1', 'ACVR2A', 'AMER1', 'APC', 'ARHGEF10', 'ARHGEF10L', 'ARID1B', 'ARID2', 'ASXL1', 'ASXL2', 'ATM', 'ATP2B3', 'ATR', 'ATRX', 'AXIN1', 'AXIN2', 'B2M', 'BAP1', 'BARD1', 'BAX', 'BAZ1A', 'BLM', 'BRCA1', 'BRCA2', 'BRIP1', 'BUB1B', 'CASP3', 'CASP8', 'CASP9', 'CBLB', 'CCNC', 'CDC73', 'CDH1', 'CDH10', 'CDK12', 'CDKN1B', 'CDKN2A', 'CDKN2C', 'CEBPA', 'CHD2', 'CHEK2', 'CNOT3', 'CNTNAP2', 'CPEB3', 'CSMD3', 'CTCF', 'CUL3', 'CYLD', 'DDX3X', 'DICER1', 'DNM2', 'DNMT3A', 'DROSHA', 'EED', 'ELF3', 'ERCC2', 'ERCC3', 'ERCC4', 'ERCC5', 'ETNK1', 'EXT2', 'FANCA', 'FANCC', 'FANCD2', 'FANCE', 'FANCF', 'FANCG', 'FAS', 'FAT1', 'FAT4', 'FBLN2', 'FBXO11', 'FBXW7', 'FEN1', 'FH', 'FLCN', 'GPC5', 'GRIN2A', 'HNF1A', 'ID3', 'IGF2BP2', 'KDM5C', 'KEAP1', 'KLF6', 'KMT2C', 'LARP4B', 'LATS1', 'LATS2', 'LEPROTL1', 'LRP1B', 'LZTR1', 'MAX', 'MED12', 'MEN1', 'MGMT', 'MLH1', 'MSH2', 'MSH6', 'MUTYH', 'N4BP2', 'NBN', 'NCOR1', 'NCOR2', 'NF2', 'NFKBIE', 'NTHL1', 'PALB2', 'PBRM1', 'PHF6', 'PHOX2B', 'PIK3R1', 'PMS2', 'POLD1', 'POLE', 'POLG', 'POT1', 'PPP2R1A', 'PPP6C', 'PRDM1', 'PRDM2', 'PRF1', 'PTCH1', 'PTEN', 'PTPN13', 'PTPN6', 'PTPRB', 'PTPRC', 'PTPRD', 'PTPRT', 'RAD17', 'RB1', 'RBM10', 'RFWD3', 'RNF43', 'ROBO2', 'RPL10', 'RPL5', 'SBDS', 'SDHA', 'SDHAF2', 'SDHB', 'SDHC', 'SDHD', 'SETD1B', 'SETD2', 'SFRP4', 'SH2B3', 'SIRPA', 'SMAD2', 'SMAD3', 'SMAD4', 'SMARCA4', 'SMARCB1', 'SMARCD1', 'SMARCE1', 'SMC1A', 'SOCS1', 'SOX21', 'SPEN', 'SPOP', 'STAG1', 'STAG2', 'STK11', 'SUFU', 'TENT5C', 'TET2', 'TGFBR2', 'TMEM127', 'TNFAIP3', 'TNFRSF14', 'TRAF7', 'TSC1', 'TSC2', 'USP44', 'VHL', 'WNK2', 'WRN', 'XPA', 'XPC', 'ZFHX3', 'ZMYM3', 'ZNRF3', 'ZRSR2', 'ARHGAP26', 'ARHGEF12', 'ARID1A', 'BCL10', 'BCOR', 'BTG1', 'CAMTA1', 'CARS', 'CBFA2T3', 'CBFB', 'CCDC6', 'CCNB1IP1', 'CD274', 'CDH11', 'CDX2', 'CIITA', 'CLTC', 'CLTCL1', 'CNBP', 'CREB3L1', 'DDX10', 'EBF1', 'EIF3E', 'ELL', 'EP300', 'EPS15', 'ETV6', 'EXT1', 'FHIT', 'FUS', 'IKZF1', 'KAT6B', 'KNL1', 'LRIG3', 'MLF1', 'MYH9', 'NAB2', 'NCOA4', 'NDRG1', 'NF1', 'NRG1', 'PATZ1', 'PER1', 'PML', 'PPARG', 'PTPRK', 'RAD51B', 'RANBP2', 'RHOH', 'RMI2', 'RPL22', 'RSPO2', 'SFPQ', 'SLC34A2', 'TPM3', 'TRIM33', 'WIF1', 'YWHAE', 'ZBTB16', 'ZNF331'],
                             'og': ['A1CF', 'ACVR1', 'AKT1', 'AKT2', 'AKT3', 'AR', 'ARAF', 'ARHGAP5', 'BCL2L12', 'CACNA1D', 'CALR', 'CARD11', 'CCNE1', 'CCR4', 'CCR7', 'CD28', 'CD79A', 'CD79B', 'CDH17', 'CDK4', 'CHD4', 'CSF1R', 'CSF3R', 'CTNNA2', 'CTNND2', 'CXCR4', 'CYSLTR2', 'DDR2', 'DGCR8', 'EGFR', 'ERBB3', 'FGFR4', 'FLT3', 'FLT4', 'FOXA1', 'FUBP1', 'GATA2', 'GNA11', 'GNAQ', 'GNAS', 'GRM3', 'H3F3A', 'H3F3B', 'HIF1A', 'HIST1H3B', 'HRAS', 'IDH1', 'IDH2', 'IKBKB', 'IL6ST', 'IL7R', 'JAK3', 'JUN', 'KAT7', 'KCNJ5', 'KDR', 'KIT', 'KNSTRN', 'KRAS', 'MACC1', 'MAP2K1', 'MAP2K2', 'MAPK1', 'MDM2', 'MDM4', 'MET', 'MITF', 'MPL', 'MTOR', 'MUC16', 'MUC4', 'MYCL', 'MYCN', 'MYD88', 'MYOD1', 'NRAS', 'NT5C2', 'PIK3CA', 'PIK3CB', 'PPM1D', 'PREX2', 'PRKACA', 'PTPN11', 'RAC1', 'REL', 'SALL4', 'SETDB1', 'SF3B1', 'SGK1', 'SIX1', 'SIX2', 'SKI', 'SMO', 'SOX2', 'SRC', 'SRSF2', 'STAT3', 'TNC', 'TRRAP', 'TSHR', 'U2AF1', 'UBR5', 'USP8', 'WAS', 'XPO1', 'ZEB1', 'ABL1', 'ABL2', 'ACKR3', 'AFDN', 'AFF3', 'AFF4', 'ALK', 'ATF1', 'BCL11A', 'BCL2', 'BCL3', 'BCL6', 'BCL9', 'BIRC6', 'BRAF', 'BRD3', 'BRD4', 'CCND1', 'CCND2', 'CCND3', 'CD74', 'CDK6', 'CHST11', 'CREB1', 'CREB3L2', 'CRLF2', 'CRTC1', 'CTNNB1', 'DDIT3', 'DDX5', 'DDX6', 'DEK', 'ELK4', 'ERBB2', 'ERG', 'ETV1', 'ETV4', 'ETV5', 'EWSR1', 'FCGR2B', 'FCRL4', 'FEV', 'FGFR1', 'FGFR2', 'FGFR3', 'FLI1', 'FOXP1', 'FOXR1', 'FSTL3', 'GLI1', 'HEY1', 'HIP1', 'HLF', 'HMGA1', 'HMGA2', 'HNRNPA2B1', 'HOXA13', 'HOXC11', 'HOXC13', 'HOXD11', 'HOXD13', 'JAK2', 'KAT6A', 'KDM5A', 'KMT2A', 'LCK', 'LMO1', 'LMO2', 'LPP', 'LYL1', 'MAF', 'MAFB', 'MALT1', 'MAML2', 'MECOM', 'MLLT10', 'MN1', 'MSI2', 'MTCP1', 'MYB', 'MYC', 'NCOA2', 'NFATC2', 'NPM1', 'NR4A3', 'NSD2', 'NSD3', 'NTRK3', 'NUP98', 'NUTM1', 'OLIG2', 'P2RY8', 'PAX3', 'PBX1', 'PDCD1LG2', 'PDGFB', 'PDGFRA', 'PDGFRB', 'PIM1', 'PLAG1', 'PLCG1', 'POU2AF1', 'POU5F1', 'PRDM16', 'PSIP1', 'RAF1', 'RAP1GDS1', 'RARA', 'RET', 'ROS1', 'RSPO3', 'SET', 'SETBP1', 'SH3GL1', 'SND1', 'SRSF3', 'SSX1', 'SSX2', 'SSX4', 'STAT6', 'STIL', 'SYK', 'TAF15', 'TAL1', 'TAL2', 'TCF7L2', 'TCL1A', 'TEC', 'TFE3', 'TFEB', 'TLX1', 'TLX3', 'TNFRSF17', 'TRIM27', 'USP6', 'WWTR1', 'ZNF521'],
                             'og_tsg': ['APOBEC3B', 'ATP1A1', 'BCL9L', 'BCORL1', 'BMPR1A', 'BTK', 'CBLC', 'CDKN1A', 'CUX1', 'DAXX', 'DDB2', 'EPAS1', 'ERBB4', 'EZH2', 'FES', 'FOXL2', 'GATA1', 'GATA3', 'GPC3', 'IRS4', 'JAK1', 'KDM6A', 'KLF4', 'KMT2D', 'LEF1', 'MAP2K4', 'MAP3K1', 'MAP3K13', 'NFE2L2', 'NKX2-1', 'NOTCH2', 'PABPC1', 'POLQ', 'PTK6', 'QKI', 'RAD21', 'RECQL4', 'RHOA', 'TBX3', 'TERT', 'TP63', 'ARNT', 'BCL11B', 'BIRC3', 'CBL', 'CIC', 'CREBBP', 'ELF4', 'ESR1', 'FOXO1', 'FOXO3', 'FOXO4', 'HOXA11', 'HOXA9', 'IRF4', 'MALAT1', 'MRTFA', 'NFKB2', 'NOTCH1', 'NTRK1', 'PAX5', 'PRKAR1A', 'RUNX1', 'RUNX1T1', 'STAT5B', 'SUZ12', 'TBL1XR1', 'TCF3', 'TET1', 'TP53', 'TRIM24', 'WT1'],
                             'unknown': ['ANK1', 'BCLAF1', 'BMP5', 'CD209', 'CNBD1', 'CRNKL1', 'CTNND1', 'CYP2C8', 'DCAF12L2', 'DCC', 'ECT2L', 'EIF1AX', 'EPHA3', 'EPHA7', 'FAM135B', 'FAM47C', 'FAT3', 'FKBP9', 'FLNA', 'ISX', 'ITGAV', 'MB21D2', 'NBEA', 'PCBP1', 'PMS1', 'PRKCB', 'PRPF40B', 'RGPD3', 'RGS7', 'ZNF429', 'ZNF479']
                             }

###############################################################################
#                                  #MAIN#                                     #
###############################################################################
if __name__ == '__main__':
    # dir organization
    timestr = time.strftime("%Y%m%d-%H%M%S")
    DIR_PATH = os.path.join(OUTPUT_DIR_PATH, "work_dataframe_directory_%s" % timestr)
    os.makedirs(DIR_PATH)
    TABLE_FILE_FORMAT = os.path.join(DIR_PATH, "dataframe_%s_%s_" + timestr + ".csv")

    # list of genes to include in the dataframe
    gene_names_list = ["EGFR", "TP53", "BRAF", "KRAS", "IDH1", "APC"]

    # list of genes that the requested transcripts is MANE transcript rather than COSMIC selection
    genes_MANE_transcripts_list = []

    # Each gene transcripts according to COSMIC https://cancer.sanger.ac.uk/cosmic/download
    cosmic_transcripts_fn = os.path.join(RAW_DATA_PATH, "CosmicTranscripts.tsv")
    cosmic_transcripts_df = pd.read_csv(cosmic_transcripts_fn, delimiter='\t')
    cosmic_transcripts_dic = dict(zip(cosmic_transcripts_df['Gene_NAME'], cosmic_transcripts_df['Transcript ID']))
    main_cosmic_transcript_dic = {}
    for gene_name in gene_names_list:
        if gene_name in cosmic_transcripts_dic:
            main_cosmic_transcript_dic[gene_name] = cosmic_transcripts_dic[gene_name][
                                                    :cosmic_transcripts_dic[gene_name].find('.')]

    # Each gene transcripts according to ensembl https://www.ensembl.org/biomart
    ensembl_transcripts_fn = os.path.join(RAW_DATA_PATH, "EnsemblTranscripts.txt")
    ensembl_transcripts_df = pd.read_csv(ensembl_transcripts_fn, delimiter='\t')
    ensembl_transcripts_df = ensembl_transcripts_df.loc[(ensembl_transcripts_df['Gene name'].isin(gene_names_list))
                                                        & (~ensembl_transcripts_df['Transcript stable ID'].isnull()) &
                                                        (~ensembl_transcripts_df['Protein stable ID'].isnull())]
    ensembl_transcripts_df = ensembl_transcripts_df.groupby('Gene name').agg(list)
    ensembl_transcripts_df.columns = ensembl_transcripts_df.columns.get_level_values(0)
    ensembl_transcripts_df.reset_index(level=0, inplace=True)
    ensembl_transcripts_dic = ensembl_transcripts_df.set_index('Gene name').T.to_dict('dict')

    # Select main transcript to all genes of interest according to following criteria:
    # 1. For list of selected genes [genes_MANE_transcripts_list] - select main transcripts as MANE transcript
    # 2. For genes with main transcript selected in COSMIC - select main transcripts as chosen by COSMIC
    # 3. For genes without main transcript selected in COSMIC - if only one known transcripts select him,
    #    otherwise select MANE transcript
    selected_transcripts = {}
    for gene_name in ensembl_transcripts_dic:
        flag = True
        list_of_transcripts_per_gene_name = list(zip(ensembl_transcripts_dic[gene_name]['Transcript stable ID'],
                                                    ensembl_transcripts_dic[gene_name]['Protein stable ID'],
                                                    ensembl_transcripts_dic[gene_name][
                                                        'RefSeq match transcript (MANE Select)']))
        if gene_name in main_cosmic_transcript_dic:
            if main_cosmic_transcript_dic[gene_name] in ensembl_transcripts_dic[gene_name]['Transcript stable ID']:
                for transcript_id, protein_id, is_mane in list_of_transcripts_per_gene_name:
                    if transcript_id == main_cosmic_transcript_dic[gene_name]:
                        selected_transcripts[gene_name] = [transcript_id, protein_id, is_mane]
                        flag = False
        if flag or (gene_name in genes_MANE_transcripts_list):
            if len(ensembl_transcripts_dic[gene_name]['Transcript stable ID']) == 1:
                for transcript_id, protein_id, is_mane in list_of_transcripts_per_gene_name:
                    selected_transcripts[gene_name] = [transcript_id, protein_id, is_mane]
            elif len(list(set(ensembl_transcripts_dic[gene_name]['RefSeq match transcript (MANE Select)']))) > 1:
                for transcript_id, protein_id, is_mane in list_of_transcripts_per_gene_name:
                    if 'NM_' in str(is_mane):
                        selected_transcripts[gene_name] = [transcript_id, protein_id, is_mane]
            else:
                selected_transcripts[gene_name] = ['No', 'No', 'No']

    # Create combined dataframe of requested genes according to selected transcript
    groupby_list = []
    for gene_name in gene_names_list:
        if gene_name in selected_transcripts:
            df_groupby_list = glob.glob(os.path.join(RAW_DATA_PATH, "dbNSFP_CSD_per_gene",
                                                     "dataframe_%s_%s*_dbnsfp_mutagene_enriched_GroupBy_*" % (
                                                     gene_name, selected_transcripts[gene_name][0])))
            if len(df_groupby_list) == 1:
                tmp_df = pd.read_csv(df_groupby_list[0], low_memory=False, na_values=['.'])
                tmp_df.columns = [i[:(i.find('_EN'))] if i.find('_EN') != -1 else i for i in list(tmp_df.columns)]
                groupby_list.append(tmp_df)
    work_df = pd.concat(groupby_list)

    # Generate amino acid change column with one letter for each amino acid
    work_df['aapos'] = work_df['aapos'].astype('int64')
    work_df['HGVSp_VEP_1L'] = work_df.apply(lambda row:'%s%s%s'%(row['aaref'], int(row['aapos']), row['aaalt']), axis=1)

    # Generate binomial test with healthy population filter
    work_df['SUM_with_healthy_population_filter'] = np.where(work_df['AlleleCount_whole_no_cancer'] > 0,
                                               work_df['SUM_4DBs_somatic'], work_df['SUM_4DBs'])
    work_df['Binomial_test_with_healthy_population_filter'] = work_df.apply(
        lambda x: stats.binom_test(x['SUM_with_healthy_population_filter'], x['Samples_Number'],
                                   x['MutaGene.mutability'] / 10 ** 6, alternative='greater'), axis=1)

    # Generate FDR correction for Binomial_test_with_healthy_population_filter
    work_df['Binomial_test_with_healthy_population_filter_fdr'] = \
        statsmodels.stats.multitest.fdrcorrection(work_df['Binomial_test_with_healthy_population_filter'],
                                                  alpha=0.1)[1]
    # Generate TVA (Tumor Variant Amplitude) with healthy population filter
    work_df['TVA'] = np.log10(work_df['SUM_with_healthy_population_filter'] / (
                work_df['Samples_Number'] * (work_df['MutaGene.mutability'] / 10 ** 6)))

    # Generate ClinVar summarized clinical significance
    work_df['clinvar_edited'] = np.where(work_df['clinvar_clnsig'].str.contains(
        'Pathogenic|Likely_pathogenic|drug_response', na=False), 'Pathogenic/Likely_pathogenic',
        np.where(work_df['clinvar_clnsig'].str.contains('Conflicting_interpretations_of_pathogenicity', na=False),
                     'Conflicting',
                 np.where(work_df['clinvar_clnsig'].str.contains('Benign', na=False), 'Benign/Likely_benign',
                          np.where(work_df['clinvar_clnsig'].str.contains('Likely_benign', na=False),
                                   'Benign/Likely_benign',
                                   np.where(work_df['clinvar_clnsig'].str.contains('Uncertain_significance', na=False),
                                      'Uncertain_significance', 'not_provided')))))

    # Generate Role in cancer according to COSMIC
    work_df['RoleInCancer'] =  np.where(work_df['genename'].isin(COSMIC_ROLE_IN_CANCER_DIC['fusion']), 'Fusion',
             np.where(work_df['genename'].isin(COSMIC_ROLE_IN_CANCER_DIC['tsg']), 'TSG',
                      np.where(work_df['genename'].isin(COSMIC_ROLE_IN_CANCER_DIC['og']), 'Oncogene',
                               np.where(work_df['genename'].isin(COSMIC_ROLE_IN_CANCER_DIC['og_tsg']), 'TSG\Oncogene',
                                                                    'Unknown'))))

    # Generate driver prediction according to binomial test and healthy population information
    adjusted_pvalue_cutoff = 0.1
    af_cutoff = 1 * (10 ** -4)
    edge_pval = 1 * (10 ** -50)
    work_df['Driver'] = np.where(
        (work_df['Binomial_test_with_healthy_population_filter_fdr'] < adjusted_pvalue_cutoff) & (
                    (work_df['Binomial_test_with_healthy_population_filter_fdr'] < edge_pval) | (
                        work_df['AlleleFrequency_whole_no_cancer'] < af_cutoff)), 1, 0)

    # Save csv tables
    columns_to_drop = ['SUM_4DBs_binom_test_greater','SUM_4DBs_binom_test_greater_log','SUM_4DBs_somatic_binom_test_greater','SUM_4DBs_somatic_binom_test_greater_log','AlleleFrequency_whole_grouped','AlleleFrequency_whole_no_cancer_grouped','SUM_4DBs_nonSomatic_AFW_binom_test_greater','SUM_4DBs_nonSomatic_AFW_binom_test_greater_log','SUM_4DBs_AFW_mutagene_binom_test_greater','SUM_4DBs_AFW_mutagene_binom_test_greater_log','CancerType_Counter','CancerType']

    catalog_fn = TABLE_FILE_FORMAT % ('Driver_FDR', 'un_%s_af_ab_%s' % (adjusted_pvalue_cutoff, af_cutoff))
    work_df[(work_df['Driver'] == 1)].drop(columns_to_drop, axis=1).to_csv(catalog_fn, index=False)
