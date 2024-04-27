import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Do not print the plot

# Original GLDS120 dataset's column names ordered
GLDS120_features = ['gene_id',
                    'Atha_Col-0_root_FLT_Alight_Rep1_GSM2493777_Day13',
                    'Atha_Col-0_root_FLT_Alight_Rep2_GSM2493778_Day13',
                    'Atha_Col-0_root_FLT_Alight_Rep3_GSM2493779_Day13',
                    'Atha_Col-0-PhyD_root_FLT_Alight_Rep1_GSM2493783_Day13',
                    'Atha_Col-0-PhyD_root_FLT_Alight_Rep2_GSM2493784_Day13',
                    'Atha_Col-0-PhyD_root_FLT_Alight_Rep3_GSM2493785_Day13',
                    'Atha_Ws_root_FLT_Alight_Rep1_GSM2493780_Day13',
                    'Atha_Ws_root_FLT_Alight_Rep2_GSM2493781_Day13',
                    'Atha_Ws_root_FLT_Alight_Rep3_GSM2493782_Day13',
                    'Atha_Col-0_root_FLT_dark_Rep1_GSM2493786_Day13',
                    'Atha_Col-0_root_FLT_dark_Rep2_GSM2493787_Day13',
                    'Atha_Col-0_root_FLT_dark_Rep3_GSM2493788_Day13',
                    'Atha_Col-0-PhyD_root_FLT_dark_Rep1_GSM2493792_Day13',
                    'Atha_Col-0-PhyD_root_FLT_dark_Rep2_GSM2493793_Day13',
                    'Atha_Col-0-PhyD_root_FLT_dark_Rep3_GSM2493794_Day13',
                    'Atha_Ws_root_FLT_dark_Rep1_GSM2493789_Day13',
                    'Atha_Ws_root_FLT_dark_Rep2_GSM2493790_Day13',
                    'Atha_Ws_root_FLT_dark_Rep3_GSM2493791_Day13',
                    'Atha_Col-0_root_GC_Alight_Rep1_GSM2493759_Day13',
                    'Atha_Col-0_root_GC_Alight_Rep2_GSM2493760_Day13',
                    'Atha_Col-0_root_GC_Alight_Rep3_GSM2493761_Day13',
                    'Atha_Col-0-PhyD_root_GC_Alight_Rep1_GSM2493765_Day13',
                    'Atha_Col-0-PhyD_root_GC_Alight_Rep2_GSM2493766_Day13',
                    'Atha_Col-0-PhyD_root_GC_Alight_Rep3_GSM2493767_Day13',
                    'Atha_Ws_root_GC_Alight_Rep1_GSM2493762_Day13',
                    'Atha_Ws_root_GC_Alight_Rep2_GSM2493763_Day13',
                    'Atha_Ws_root_GC_Alight_Rep3_GSM2493764_Day13',
                    'Atha_Col-0_root_GC_dark_Rep1_GSM2493768_Day13',
                    'Atha_Col-0_root_GC_dark_Rep2_GSM2493769_Day13',
                    'Atha_Col-0_root_GC_dark_Rep3_GSM2493770_Day13',
                    'Atha_Col-0-PhyD_root_GC_dark_Rep1_GSM2493774_Day13',
                    'Atha_Col-0-PhyD_root_GC_dark_Rep2_GSM2493775_Day13',
                    'Atha_Col-0-PhyD_root_GC_dark_Rep3_GSM2493776_Day13',
                    'Atha_Ws_root_GC_dark_Rep1_GSM2493771_Day13',
                    'Atha_Ws_root_GC_dark_Rep2_GSM2493772_Day13',
                    'Atha_Ws_root_GC_dark_Rep3_GSM2493773_Day13'
                    ]
# Change column names
new_col_names = ['Rep1_c_l', 'Rep2_c_l', 'Rep3_c_l',
                 'Rep1_p_l', 'Rep2_p_l', 'Rep3_p_l',
                 'Rep1_w_l', 'Rep2_w_l', 'Rep3_w_l',
                 'Rep1_c_d', 'Rep2_c_d', 'Rep3_c_d',
                 'Rep1_p_d', 'Rep2_p_d', 'Rep3_p_d',
                 'Rep1_w_d', 'Rep2_w_d', 'Rep3_w_d']


def restructure_data(source_df):
    # Order df
    nc_all = source_df[GLDS120_features]
    # Get repeated gene_id df
    gene_id_df = pd.DataFrame(np.repeat(nc_all['gene_id'].values, 2, axis=0), columns=['gene_id'])
    # Create Location variable
    LC = pd.DataFrame({"Location": ['FLT', 'GC']})
    LC = pd.concat([LC] * len(nc_all)).reset_index(drop=True)

    # Normalized counts (fpkm) data without id
    nc_wo_id = nc_all.drop(columns=['gene_id'])
    # Restructure dataset
    new_nc = pd.DataFrame(nc_wo_id.values.reshape(-1, 2, 18).reshape(len(LC), 18), columns=new_col_names)
    new_nc = pd.concat([gene_id_df, new_nc], axis=1)
    new_nc['Location'] = LC

    return new_nc


def concat_df(representation, df, dimension):
    # concat representation coordinates with the nc_df
    if dimension == 2:
        coordinates_df = pd.DataFrame(representation, columns=['x', 'y'])
    elif dimension == 3:
        coordinates_df = pd.DataFrame(representation, columns=['x', 'y', 'z'])
    else:
        raise TypeError("Choose 2 or 3 for dimension")
    coordinates_df['gene_id'] = df['gene_id']
    coordinates_df['Location'] = df['Location']

    return coordinates_df


def preprocessing_kmeans(loc_df, location):
    # Get data representation via pca
    # PCA in 3-dimension for 3d-plot
    loc_df_pca = PCA(n_components=3).fit_transform(loc_df.iloc[:, 1:-1])
    # Elbow method for k-means
    # Initialize kmeans parameters
    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 10,
        "random_state": 1,
    }
    # Create list to hold SSE values for each k
    sse_lst = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        # You can run it on the raw data without dimensionality reduction but it detects same outliers
        kmeans.fit(loc_df_pca)
        sse_lst.append(kmeans.inertia_)
    # Visualize results
    plt.plot(range(1, 11), sse_lst, markersize=3, marker='o')
    plt.xticks(range(1, 11))
    plt.title("Elbow method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.savefig('elbow_method_plot_' + location + '.png')

    # After elbow method we set num_cluster = 5 for FLT and = 4 for GC
    num_cluster = 5 if location == 'FLT' else 4
    # Cluster with tuned number of cluster
    model = KMeans(n_clusters=num_cluster, **kmeans_kwargs)
    # Predict cluster and make df
    y_clusters = model.fit_predict(loc_df_pca)
    loc_df_pca = pd.DataFrame(loc_df_pca, columns=['x', 'y', 'z'])
    loc_df_pca['cluster'] = y_clusters
    loc_df_pca = loc_df_pca.astype({'cluster': 'str'})
    loc_df_pca['gene_id'] = loc_df['gene_id']

    # # export .csv for the preprocessing data, explore how the clusters are distributed
    # loc_df_pca.to_csv('outlier_detection_df_' + location + '.csv', index=False)
    # Detected outliers for the GLDS-120 dataset
    outlier_gene_id =["AT3G41768", "ATMG00020", "AT1G07590"]
    # Clean df
    clean_df = loc_df.drop(loc_df[loc_df['gene_id'].isin(outlier_gene_id)].index).reset_index(drop=True)

    return clean_df


def tsne4viz(nc_df, representation, dim):
    # t-SNE, 2-dimension # change / tune perplexity as need
    rep_viz = TSNE(n_components=dim, perplexity=75, random_state=1996, n_jobs=-1,
                   learning_rate='auto').fit_transform(representation)
    # Get full pca_df
    rep_viz_df = concat_df(rep_viz, nc_df, dimension=dim)

    return rep_viz_df
