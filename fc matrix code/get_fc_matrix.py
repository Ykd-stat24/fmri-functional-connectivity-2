# Script to extract mean functional connectivity matrix of preprocessed rsfmri parcellated data.

from pathlib import Path

import h5py
import pandas as pd
import numpy as np

def get_ts(file_path, labels=False):
    """
    Get the func timeseries data of a .h5 file. 

    Returns a DataFrame where each column corresponds to a region.
    """
    
    with h5py.File(file_path) as fdata:

        # only load the first 400 regions, which correspond to the Schaefer atlas
        schaefer_data = fdata['dataset'][:400,:]

    schaefer_data = pd.DataFrame(schaefer_data)
    
    if labels:
        schaefer_data.index = labels
    
    schaefer_data = schaefer_data.transpose()

    return schaefer_data

def get_fc_matrix(ts):
    fc_matrix = ts.corr()
    return fc_matrix

def get_subject_avg_fc(fc_list, labels=False):
    """
    Get the average fc matrix of a subject.
    """

    fc_array_concatenated = np.stack([np.array(fc_array) for fc_array in fc_list], axis=2)

    fc_array_averaged = fc_array_concatenated.mean(axis=2)

    fc_array_averaged = pd.DataFrame(fc_array_averaged)

    if labels:
        fc_array_averaged.columns = labels
        fc_array_averaged.index = labels

    return fc_array_averaged


def subject_pipeline(subject_path, labels):
    """
    Whole pipeline for extracting mean fc matrix of one subject.
    """

    fc_list = []
    for rs_file in subject_path.glob('*task-rest*'):
        ts = get_ts(rs_file, labels)
        fc = get_fc_matrix(ts)
        fc_list.append(fc)

    fc_averaged = get_subject_avg_fc(fc_list, labels)

    return fc_averaged

def get_bids_subject_name(subject_path):
    return 'sub-' + subject_path.name.replace('_', '') + '_task-rest_desc-FCmatrix.csv'



def main():

    preprocessed_path = Path('inputs/ds005237/fMRI_timeseries_clean_denoised_GSR_parcellated/')
    derivatives_path = Path('derivatives/mean_fc_matrix')
    derivatives_path.mkdir(exist_ok=True, parents=True)

    # atlas labels
    labels = pd.read_csv("inputs/schaefer_labels.csv", header=None)[0].to_list()

    for subject_path in preprocessed_path.glob('*'):

        print(f"Extracting data from: {subject_path.name}")

        # get the output file name
        subject_fc_file = get_bids_subject_name(subject_path)
        subject_fc_file = derivatives_path / subject_fc_file

        # get the mean fc matrix
        subject_mean_fc_matrix = subject_pipeline(subject_path, labels)

        subject_mean_fc_matrix.to_csv(subject_fc_file, index=True)


    return None

if __name__ == '__main__':
    main()