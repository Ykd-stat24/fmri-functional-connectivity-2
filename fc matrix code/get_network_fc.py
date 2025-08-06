from pathlib import Path
import re
import pandas as pd
import numpy as np


def matrix2long(fc_matrix):
    """
    Transform a FC matrix DataFrame to a long format DataFrame.
    """

    triu_idx = np.triu_indices_from(fc_matrix, k=1)
    # Extract node labels
    node1_labels = fc_matrix.index[triu_idx[0]]
    node2_labels = fc_matrix.columns[triu_idx[1]]
    fc_values = fc_matrix.values[triu_idx]

    # Create a long-form DataFrame
    fc_long = pd.DataFrame({
        'node1': node1_labels,
        'node2': node2_labels,
        'fc': fc_values
    })

    node1 = (fc_long
    ['node1']
    .str.extract(r'17Networks_([R,L]H)_([a-z,A-Z,0-9]+)_(\w+)')
    .replace('', pd.NA)
    .rename({0:'node1_hemisphere', 1:'node1_network', 2:'node1_label'}, axis=1)
    )

    node2 = (fc_long
    ['node2']
    .str.extract(r'17Networks_([R,L]H)_([a-z,A-Z,0-9]+)_(\w+)')
    .replace('', pd.NA)
    .rename({0:'node2_hemisphere', 1:'node2_network', 2:'node2_label'}, axis=1)
    )

    fc_long = (fc_long
        .merge(node1, left_index=True, right_index=True)
        .merge(node2, left_index=True, right_index=True)
        )
    

    return fc_long


def add_network_edges(fc_long):
    """
    Add network edges to a fc long-format DataFrame.
    """

    networks = fc_long['node1_network'].dropna().unique().tolist()

    # node 1 and node 2 contain permutations, we have to replace to combinations. 
    network_permutations = {}
    for network1 in networks:
        for network2 in networks:
            # make sure order doesn't matter (combination)
            networks_sorted = pd.Series([network1, network2]).sort_values().to_list()
            network_combination = f'{networks_sorted[0]}-{networks_sorted[1]}'
            # order does matter (permutation)
            network_permutation = f'{network1}-{network2}'
            # dictionary permutation-combination
            network_permutations[network_permutation] = network_combination

    fc_long.loc[:,'network_permutation'] = fc_long['node1_network'] + '-' + fc_long['node2_network']
    # replace permutation to corresponding combination (network edge)
    fc_long['network_connection'] = fc_long['network_permutation'].replace(network_permutations)
    # is this a within-network edge?
    fc_long.loc[:, 'within_network'] = fc_long['node1_network'] == fc_long['node2_network']

    return fc_long


def get_network_fc(fc_long):
    """
    Get the mean functional connectivity of all between and within network edges.
    Make sure your long FC DataFrame includes the labels of network edges.
    """

    network_avg_fc = (fc_long
        .groupby('network_connection')
        ['fc']
        .mean()
        .to_frame()
        .reset_index()
        .merge(fc_long[['network_connection', 'within_network']].drop_duplicates()))
    
    return network_avg_fc


def network_fc_pipeline(fc_matrix_path):
    """
    Run the whole pipeline to extract mean network functional connectivity.
    """

    fc_matrix = pd.read_csv(fc_matrix_path, index_col=0)

    fc_long = matrix2long(fc_matrix)

    fc_long = add_network_edges(fc_long)

    fc_network = get_network_fc(fc_long)

    return fc_network


def main():

    mean_fc_matrix_dir = Path('derivatives/mean_fc_matrix')
    output_dir = Path('derivatives/network_fc')
    output_dir.mkdir(exist_ok=True)

    for mean_fc_matrix_path in mean_fc_matrix_dir.glob('*_task-rest_desc-FCmatrix.csv'):

        # subject output filename
        output_filename = mean_fc_matrix_path.name.replace('desc-FCmatrix.csv', 'desc-FCnetwork.csv')
        subject = re.search('sub-\w+(?=_)', output_filename).group()
        output_filename = output_dir / output_filename

        print(f"Extracting data of {subject}")

        # get derivative
        fc_network = network_fc_pipeline(mean_fc_matrix_path)

        # write
        fc_network.to_csv(output_filename, index=False)

if __name__ == '__main__':
    main()