import os

import anndata
import scanpy as sc
import scgen
from scipy import sparse
import numpy as np


def test_train_whole_data_one_celltype_out(data_name="pbmc",
                                           z_dim=50,
                                           alpha=0.1,
                                           n_epochs=1000,
                                           batch_size=32,
                                           dropout_rate=0.25,
                                           learning_rate=0.001,
                                           condition_key="condition",
                                           cell_type_to_train=None):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train_pbmc.h5ad")
        valid = sc.read("../data/valid_pbmc.h5ad")
    elif data_name == "hpoly":
        stim_key = "Hpoly.Day10"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/train_hpoly.h5ad")
        valid = sc.read("../data/valid_hpoly.h5ad")
    elif data_name == "salmonella":
        stim_key = "Salmonella"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/train_salmonella.h5ad")
        valid = sc.read("../data/valid_salmonella.h5ad")
    elif data_name == "species":
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
        train = sc.read("../data/train_species.h5ad")
        valid = sc.read("../data/valid_species.h5ad")

    for cell_type in train.obs[cell_type_key].unique().tolist():
        if cell_type_to_train is not None and cell_type != cell_type_to_train:
            continue
        net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]
        net_valid_data = valid[~((valid.obs[cell_type_key] == cell_type) & (valid.obs[condition_key] == stim_key))]
        network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                                 z_dimension=z_dim,
                                 alpha=alpha,
                                 dropout_rate=dropout_rate,
                                 learning_rate=learning_rate,
                                 model_path=f"../models/scGen/{data_name}/{cell_type}/scgen")

        network.train(net_train_data, use_validation=True, valid_data=net_valid_data, n_epochs=n_epochs, batch_size=batch_size)
        network.sess.close()
        print(f"network_{cell_type} has been trained!")


def log_shape(name, obj):
    """Helper to log shapes for debugging"""
    if hasattr(obj, 'shape'):
        print(f"[SHAPE] {name}: {obj.shape}")
    elif hasattr(obj, 'X'):
        if hasattr(obj.X, 'shape'):
            print(f"[SHAPE] {name}: {obj.shape} (X: {obj.X.shape})")
        else:
            print(f"[SHAPE] {name}: {obj.shape} (X: sparse)")
    else:
        print(f"[SHAPE] {name}: {type(obj)}")


def reconstruct_whole_data(data_name="pbmc", condition_key="condition"):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train_pbmc.h5ad")
    elif data_name == "hpoly":
        stim_key = "Hpoly.Day10"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/train_hpoly.h5ad")
    elif data_name == "salmonella":
        stim_key = "Salmonella"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/train_salmonella.h5ad")
    elif data_name == "species":
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
        train = sc.read("../data/train_species.h5ad")
    elif data_name == "study":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train_study.h5ad")

    log_shape("train (full dataset)", train)

    all_data = anndata.AnnData()
    for idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
        print(f"Reconstructing for {cell_type}")
        
        # CRITICAL FIX: Create net_train_data BEFORE model initialization
        # This ensures we use the same filtered data shape as during training
        net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]
        
        log_shape("net_train_data (filtered)", net_train_data)
        
        # FIX: Use net_train_data.X.shape[1] instead of train.X.shape[1]
        # This matches what was used during training (line 46)
        network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                                 z_dimension=100,
                                 alpha=0.00005,
                                 dropout_rate=0.2,
                                 learning_rate=0.001,
                                 model_path=f"../models/scGen/{data_name}/{cell_type}/scgen")
        
        log_shape("network.x_dim", network.x_dim)
        
        network.restore_model()

        cell_type_data = train[train.obs[cell_type_key] == cell_type]
        cell_type_ctrl_data = train[((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == ctrl_key))]
        
        log_shape("cell_type_data", cell_type_data)
        log_shape("cell_type_ctrl_data", cell_type_ctrl_data)
        
        # Ensure net_train_data has same genes as model expects
        if net_train_data.X.shape[1] != network.x_dim:
            raise ValueError(
                f"Gene dimension mismatch: net_train_data has {net_train_data.X.shape[1]} genes, "
                f"but model expects {network.x_dim}. "
                f"Check if preprocessing (HVG selection, filtering) was applied inconsistently."
            )
        
        pred, delta = network.predict(adata=net_train_data,
                                      conditions={"ctrl": ctrl_key, "stim": stim_key},
                                      cell_type_key=cell_type_key,
                                      condition_key=condition_key,
                                      celltype_to_predict=cell_type)
        
        log_shape("pred (from network.predict)", pred)
        
        # Validate prediction shape matches expected
        expected_pred_genes = network.x_dim
        if pred.shape[1] != expected_pred_genes:
            raise ValueError(
                f"Prediction shape mismatch: pred has {pred.shape[1]} genes, "
                f"but model outputs {expected_pred_genes} genes."
            )
        
        # Handle gene space mapping if prediction has fewer genes than cell_type_data
        # (e.g., if training used HVG subset but reconstruction expects full gene space)
        if pred.shape[1] != cell_type_data.X.shape[1]:
            print(f"WARNING: Prediction has {pred.shape[1]} genes, but cell_type_data has {cell_type_data.X.shape[1]} genes.")
            print("Mapping prediction to full gene space...")
            
            # Create full-size array
            pred_full = np.zeros((pred.shape[0], cell_type_data.X.shape[1]))
            
            # Map predicted genes to correct positions
            # This assumes net_train_data.var_names is a subset of cell_type_data.var_names
            train_gene_idx = {gene: idx for idx, gene in enumerate(net_train_data.var_names)}
            full_gene_idx = {gene: idx for idx, gene in enumerate(cell_type_data.var_names)}
            
            # Find mapping
            mapped_count = 0
            for train_gene in net_train_data.var_names:
                if train_gene in full_gene_idx:
                    train_idx = train_gene_idx[train_gene]
                    full_idx = full_gene_idx[train_gene]
                    pred_full[:, full_idx] = pred[:, train_idx]
                    mapped_count += 1
            
            print(f"Mapped {mapped_count} genes from {pred.shape[1]} to {cell_type_data.X.shape[1]} gene space.")
            
            if mapped_count != pred.shape[1]:
                unmapped_genes = set(net_train_data.var_names) - set(full_gene_idx.keys())
                unmapped_preview = list(unmapped_genes)[:5]
                raise ValueError(
                    f"Gene mapping incomplete: only {mapped_count}/{pred.shape[1]} genes mapped. "
                    f"Unmapped genes (showing first 5): {unmapped_preview}. "
                    f"Check var_names alignment between net_train_data and cell_type_data."
                )
            
            pred = pred_full

        pred_adata = anndata.AnnData(pred, obs={condition_key: [f"{cell_type}_pred_stim"] * len(pred),
                                                cell_type_key: [cell_type] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})
        ctrl_adata = anndata.AnnData(cell_type_ctrl_data.X,
                                     obs={condition_key: [f"{cell_type}_ctrl"] * len(cell_type_ctrl_data),
                                          cell_type_key: [cell_type] * len(cell_type_ctrl_data)},
                                     var={"var_names": cell_type_ctrl_data.var_names})
        if sparse.issparse(cell_type_data.X):
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X.toarray()
        else:
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X
        real_stim_adata = anndata.AnnData(real_stim,
                                          obs={condition_key: [f"{cell_type}_real_stim"] * len(real_stim),
                                               cell_type_key: [cell_type] * len(real_stim)},
                                          var={"var_names": cell_type_data.var_names})
        if idx == 0:
            all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        else:
            all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)

        print(f"Finish Reconstructing for {cell_type}")
        network.sess.close()
    
    # Ensure output directory exists and write results
    output_path = f"../data/reconstructed/scGen/{data_name}.h5ad"
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        all_data.write_h5ad(output_path)
    except (IOError, OSError) as e:
        raise RuntimeError(f"Failed to write reconstruction output to {output_path}: {e}")


def test_train_whole_data_some_celltypes_out(data_name="pbmc",
                                             z_dim=100,
                                             alpha=0.00005,
                                             n_epochs=300,
                                             batch_size=32,
                                             dropout_rate=0.2,
                                             learning_rate=0.001,
                                             condition_key="condition",
                                             c_out=None,
                                             c_in=None):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train_pbmc.h5ad")
        valid = sc.read("../data/valid_pbmc.h5ad")

    net_train_data = scgen.data_remover(train, remain_list=c_in, remove_list=c_out,
                                        cell_type_key=cell_type_key, condition_key=condition_key)

    net_valid_data = scgen.data_remover(valid, remain_list=c_in, remove_list=c_out,
                                        cell_type_key=cell_type_key, condition_key=condition_key)

    network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                             z_dimension=z_dim,
                             alpha=alpha,
                             dropout_rate=dropout_rate,
                             learning_rate=learning_rate,
                             model_path=f"../models/scGen/pbmc/heldout/{len(c_out)}/scgen")

    network.train(net_train_data, use_validation=True, valid_data=net_valid_data, n_epochs=n_epochs, batch_size=batch_size)
    print(f"network has been trained!")
    network.sess.close()


def train_cross_study(data_name="study",
                      z_dim=100,
                      alpha=0.00005,
                      n_epochs=300,
                      batch_size=32,
                      dropout_rate=0.2,
                      learning_rate=0.001):
    train = sc.read("../data/train_study.h5ad")
    valid = sc.read("../data/valid_study.h5ad")

    net_train_data = train
    network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                             z_dimension=z_dim,
                             alpha=alpha,
                             dropout_rate=dropout_rate,
                             learning_rate=learning_rate,
                             model_path="../models/scGen/study/scgen")

    network.train(net_train_data, use_validation=True, valid_data=valid, n_epochs=n_epochs, batch_size=batch_size)
    print(f"network_{data_name} has been trained!")
    network.sess.close()


if __name__ == '__main__':
    test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
                                           dropout_rate=0.2, learning_rate=0.001)
    test_train_whole_data_one_celltype_out("hpoly", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
                                           dropout_rate=0.2, learning_rate=0.001)
    test_train_whole_data_one_celltype_out("salmonella", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
                                           dropout_rate=0.2, learning_rate=0.001)
    test_train_whole_data_one_celltype_out("species", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
                                           dropout_rate=0.2, learning_rate=0.001, cell_type_to_train="rat")
    train_cross_study("study", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
                      dropout_rate=0.2, learning_rate=0.001)
    reconstruct_whole_data("pbmc")
    reconstruct_whole_data("hpoly")
    reconstruct_whole_data("salmonella")
    reconstruct_whole_data("species")

    c_in = ['NK', 'B', 'CD14+Mono']
    c_out = ['CD4T', 'FCGR3A+Mono', 'CD8T', 'Dendritic']
    test_train_whole_data_some_celltypes_out(data_name="pbmc",
                                             z_dim=100,
                                             alpha=0.00005,
                                             n_epochs=300,
                                             batch_size=32,
                                             dropout_rate=0.2,
                                             learning_rate=0.001,
                                             condition_key="condition",
                                             c_out=c_out,
                                             c_in=c_in)
    c_in = ['CD14+Mono']
    c_out = ['CD4T', 'FCGR3A+Mono', 'CD8T', 'NK', 'B', 'Dendritic']
    test_train_whole_data_some_celltypes_out(data_name="pbmc",
                                             z_dim=100,
                                             alpha=0.00005,
                                             n_epochs=300,
                                             batch_size=32,
                                             dropout_rate=0.2,
                                             learning_rate=0.001,
                                             condition_key="condition",
                                             c_out=c_out,
                                             c_in=c_in)
    c_in = ['CD8T', 'NK', 'B', 'Dendritic', 'CD14+Mono']
    c_out = ['CD4T', 'FCGR3A+Mono']
    test_train_whole_data_some_celltypes_out(data_name="pbmc",
                                             z_dim=100,
                                             alpha=0.00005,
                                             n_epochs=300,
                                             batch_size=32,
                                             dropout_rate=0.2,
                                             learning_rate=0.001,
                                             condition_key="condition",
                                             c_out=c_out,
                                             c_in=c_in)
