import anndata
import numpy as np
import scanpy.api as sc
# from data_reader import data_reader
# from hf import *
import scgen
import scipy.sparse as sparse
from sklearn.decomposition import PCA
from scgen.file_utils import ensure_dir_for_file, get_dense_X, to_dense


# =============================== downloading training and validation files ====================================
# we do not use the validation data to apply vector arithmetics in gene expression space


def predict(pca, cd_x, hfd_x, cd_y, p_type="unbiased"):
    if p_type == "unbiased":
        eq = min(len(cd_x), len(hfd_x))
        cd_ind = np.random.choice(range(len(cd_x)), size=eq, replace=False)
        stim_ind = np.random.choice(range(len(hfd_x)), size=eq, replace=False)
    else:
        cd_ind = np.arange(0, len(cd_x))
        stim_ind = np.arange(0, len(hfd_x))
    cd = np.average(cd_x[cd_ind, :], axis=0)
    stim = np.average(hfd_x[stim_ind, :], axis=0)
    delta = stim - cd
    predicted_cells_pca = delta + cd_y
    predicted_cells = pca.inverse_transform(predicted_cells_pca)
    return predicted_cells


def reconstruct():
    train_path = "../data/train_pbmc.h5ad"
    data = sc.read(train_path)
    ctrl_key = "control"
    stim_key = "stimulated"
    all_data = anndata.AnnData()
    print(data.obs["cell_type"].unique().tolist())
    for idx, cell_type in enumerate(data.obs["cell_type"].unique().tolist()):
        pca = PCA(n_components=100)
        train = data[~((data.obs["condition"] == stim_key) & (data.obs["cell_type"] == cell_type))]
        pca.fit(get_dense_X(train))
        print(cell_type, end="\t")
        train_real_stimulated = data[data.obs["condition"] == stim_key, :]
        train_real_stimulated = train_real_stimulated[train_real_stimulated.obs["cell_type"] != cell_type]
        train_real_stimulated = scgen.util.balancer(train_real_stimulated)
        train_real_stimulated_PCA = pca.transform(train_real_stimulated.X)

        train_real_cd = data[data.obs["condition"] == ctrl_key, :]
        train_real_cd = scgen.util.balancer(train_real_cd)
        train_real_cd_PCA = pca.transform(train_real_cd.X)

        cell_type_adata = data[data.obs["cell_type"] == cell_type]
        cell_type_ctrl = cell_type_adata[cell_type_adata.obs["condition"] == ctrl_key]
        cell_type_stim = cell_type_adata[cell_type_adata.obs["condition"] == stim_key]
        # Use get_dense_X to handle views and sparse matrices
        cell_type_ctrl_PCA = pca.transform(get_dense_X(cell_type_ctrl))
        predicted_cells = predict(pca, train_real_cd_PCA, train_real_stimulated_PCA, cell_type_ctrl_PCA)
        # Convert to dense before concatenation
        all_Data = sc.AnnData(np.concatenate([get_dense_X(cell_type_ctrl), get_dense_X(cell_type_stim), predicted_cells]))
        all_Data.obs["condition"] = [f"{cell_type}_ctrl"] * cell_type_ctrl.shape[0] + [f"{cell_type}_real_stim"] * \
                                    cell_type_stim.shape[0] + \
                                    [f"{cell_type}_pred_stim"] * len(predicted_cells)
        all_Data.obs["cell_type"] = [f"{cell_type}"] * (
                cell_type_ctrl.shape[0] + cell_type_stim.shape[0] + len(predicted_cells))
        all_Data.var_names = cell_type_adata.var_names

        if idx == 0:
            all_data = all_Data
        else:
            all_data = all_data.concatenate(all_Data)
        print(cell_type)
    sc.write(ensure_dir_for_file("../data/reconstructed/PCAVecArithm/PCA_pbmc.h5ad"), all_data)


def train(data_name="pbmc", cell_type="CD4T", p_type="unbiased"):
    train_path = f"../data/train_{data_name}.h5ad"
    if data_name == "pbmc":
        ctrl_key = "control"
        stim_key = "stimulated"
        cell_type_key = "cell_type"
    elif data_name == "hpoly":
        ctrl_key = "Control"
        stim_key = "Hpoly.Day10"
        cell_type_key = "cell_label"
    elif data_name == "salmonella":
        ctrl_key = "Control"
        stim_key = "Salmonella"
        cell_type_key = "cell_label"
    data = sc.read(train_path)
    print("data has been loaded!")
    train = data[~((data.obs["condition"] == stim_key) & (data.obs[cell_type_key] == cell_type))]
    pca = PCA(n_components=100)

    pca.fit(get_dense_X(train))

    train_real_cd = train[train.obs["condition"] == "control", :]
    if p_type == "unbiased":
        train_real_cd = scgen.util.balancer(train_real_cd)
    train_real_stimulated = train[train.obs["condition"] == "stimulated", :]
    if p_type == "unbiased":
        train_real_stimulated = scgen.util.balancer(train_real_stimulated)

    # Convert to dense using utility functions
    train_real_cd = to_dense(train_real_cd)
    train_real_stimulated = to_dense(train_real_stimulated)

    train_real_stimulated_PCA = pca.transform(train_real_stimulated.X)
    train_real_cd_PCA = pca.transform(train_real_cd.X)

    adata_list = scgen.util.extractor(data, cell_type, {"ctrl": ctrl_key, "stim": stim_key})
    # Convert views to dense
    adata_list[1] = to_dense(adata_list[1])
    adata_list[2] = to_dense(adata_list[2])
    ctrl_CD4T_PCA = pca.transform(adata_list[1].X)
    predicted_cells = predict(pca, train_real_cd_PCA, train_real_stimulated_PCA, ctrl_CD4T_PCA, p_type)

    all_Data = sc.AnnData(np.concatenate([adata_list[1].X, adata_list[2].X, predicted_cells]))
    all_Data.obs["condition"] = ["ctrl"] * len(adata_list[1].X) + ["real_stim"] * len(adata_list[2].X) + \
                                ["pred_stim"] * len(predicted_cells)
    all_Data.var_names = adata_list[3].var_names
    if p_type == "unbiased":
        sc.write(ensure_dir_for_file(f"../data/reconstructed/PCAVecArithm/PCA_CD4T.h5ad"), all_Data)
    else:
        sc.write(ensure_dir_for_file(f"../data/reconstructed/PCAVecArithm/PCA_CD4T_biased.h5ad"), all_Data)


if __name__ == "__main__":
    # sc.pp.neighbors(all_Data)
    # sc.tl.umap(all_Data)
    # import matplotlib
    # import matplotlib.style
    # import matplotlib.pyplot as plt
    #
    # plt.style.use('default')
    # sc.pl.umap(all_Data, color=["condition"], frameon=False, palette=matplotlib.rcParams["axes.prop_cycle"]
    #            , save="Vec_Arith_PCA_biased.png", show=False,
    #            legend_fontsize=18, title="")
    # sc.pl.violin(all_Data, groupby='condition', keys="ISG15", save="Vec_Arith_PCA.pdf", show=False)
    train("pbmc", "CD4T", "unbiased")
    train("pbmc", "CD4T", "biased")
    reconstruct()
