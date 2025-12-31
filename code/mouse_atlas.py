import numpy as np
import scanpy as sc
from random import  shuffle
import wget
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


train_path = "../data/MouseAtlas.subset.h5ad"

if os.path.isfile(train_path):
    data = sc.read(train_path)
else:
    train_url = "https://www.dropbox.com/s/zkss8ds1pi0384p/MouseAtlas.subset.h5ad?dl=1"
    t_dl = wget.download(train_url, train_path)
    data = sc.read(train_path)




sc.settings.figdir = "../results"
model_to_use = "../models/mouse_atlas/scgen"
batch_size = 512  # Safe batch size for ml.g6e.4xlarge (48GB GPU)
train_real = data
input_matrix = data.X
ind_list = [i for i in range(input_matrix.shape[0])]
shuffle(ind_list)
train_data = input_matrix[ind_list, :]
gex_size = input_matrix.shape[1]
X_dim = gex_size
z_dim = 100
lr = 0.001
dr_rate = .2
X = tf.placeholder(tf.float32, shape=[None, X_dim],name="data")
z = tf.placeholder(tf.float32, shape=[None, z_dim],name="noise")
data_max_value = np.amax(input_matrix)
time_step = tf.placeholder(tf.int32)
size  = tf.placeholder(tf.int32)
is_training = tf.placeholder(tf.bool)
init_w = tf.keras.initializers.GlorotUniform()
regularizer = tf.keras.regularizers.l2(0.1)

def give_me_latent(data):
    latent = sess.run(z_mean,feed_dict = {X : data,size:len(data),is_training:False})
    return  latent
def avg_vector(data):

    latent =  give_me_latent(data)
    arithmatic =  np.average(latent,axis=0)
    return  arithmatic
def reconstruct(data,use_data = False):

    if(use_data):
        latent = data
    else:
        latent = give_me_latent(data)

    reconstruct = sess.run(X_hat,feed_dict = {z_mean : latent ,is_training:False})
    return  reconstruct

# =============================== Batch Normalization Workaround ======================================
# Workaround for tf.layers.batch_normalization which isn't available in TensorFlow 2.x
# This manually implements batch normalization using tf.nn functions that still work
def _work_around(scope, feature_dim, h, training):
    """
    Manual batch normalization workaround for TensorFlow 2.x compatibility.
    
    How it works:
    1. Creates trainable scale (gamma) and offset (beta) variables
    2. Computes batch mean and variance using tf.nn.moments
    3. Applies normalization: (x - mean) / sqrt(variance + epsilon) * scale + offset
    
    This is equivalent to tf.layers.batch_normalization but uses low-level TF1.x APIs
    that are still available in TF2.x via compat.v1.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        scale = tf.get_variable("scale", shape=[feature_dim], initializer=tf.ones_initializer())
        offset = tf.get_variable("offset", shape=[feature_dim], initializer=tf.zeros_initializer())
        batch_mean, batch_var = tf.nn.moments(h, axes=[0])
        return tf.nn.batch_normalization(h, batch_mean, batch_var, offset, scale, variance_epsilon=1e-5)

# =============================== Q(z|X) ======================================

def Q(X, reuse=False):
    with tf.variable_scope("gq", reuse=reuse):
        h = tf.layers.dense(inputs=X, units=800, kernel_initializer=init_w,use_bias=False,
                            kernel_regularizer=regularizer)
        h = _work_around("gq_bn_800_1", 800, h, is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h,dr_rate, training= is_training)
        h = tf.layers.dense(inputs=h, units=800, kernel_initializer=init_w, use_bias=False,
                            kernel_regularizer=regularizer)
        h = _work_around("gq_bn_800_2", 800, h, is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h,dr_rate, training= is_training)
        mean =  tf.layers.dense(inputs=h, units=z_dim, kernel_initializer=init_w)
        variance =  tf.layers.dense(inputs=h, units=z_dim, kernel_initializer=init_w)
        return mean, variance

# =============================== P(Z) ======================================
def sample_z(mu, log_var,size):
    eps = tf.random_normal(shape=[size,z_dim])
    return mu + tf.exp(log_var / 2) * eps

def sample(n_sample):
    noise = np.random.normal(0.0, 1, size=(n_sample, z_dim))
    gen_cells = sess.run(X_hat, feed_dict={z_mean: noise,is_training:False})
    return  gen_cells
# =============================== P(X|z) ======================================
def P(z,reuse=False):
    with tf.variable_scope("gp", reuse=reuse):
        h = tf.layers.dense(inputs=z,units= 800,kernel_initializer=init_w,use_bias=False,
                            kernel_regularizer=regularizer)
        h = _work_around("gp_bn_800_1", 800, h, is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h,dr_rate, training= is_training)

        h = tf.layers.dense(inputs=h, units=800, kernel_initializer=init_w,use_bias=False,
                            kernel_regularizer=regularizer)
        h = _work_around("gp_bn_800_2", 800, h, is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h,dr_rate, training= is_training)
        h = tf.layers.dense(inputs=h, units=X_dim, kernel_initializer=init_w, use_bias=True)
        h = tf.nn.relu(h)
        return h
mean, variance = Q(X)
z_mean = sample_z(mean,variance,size)
X_hat = P(z_mean)
# =============================== loss ====================================
kl_loss = 0.5 * tf.reduce_sum(tf.exp(variance) + mean**2 - 1. - variance, 1)
recon_loss = 0.5*tf.reduce_sum(tf.square((X-X_hat)), 1)
vae_loss = tf.reduce_mean(recon_loss + 0.00005 * kl_loss)
recon_loss_summary = tf.summary.scalar('REC', tf.reduce_mean(recon_loss))
vae_loss_summary = tf.summary.scalar('VAE', vae_loss)
t_vars = tf.trainable_variables()
g_lrate = tf.placeholder(tf.float32, shape=[])
global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    Solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(vae_loss)
# Configure GPU settings
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Allow GPU memory to grow dynamically
# Optionally set which GPU to use (uncomment and set if you have multiple GPUs):
# config.gpu_options.visible_device_list = "0"  # Use GPU 0
sess = tf.InteractiveSession(config=config)
saver = tf.train.Saver(max_to_keep=1)
init = tf.global_variables_initializer().run()

def train(n_epochs, full_training=True, initial_run=True):
    if initial_run:
        print("Initial run")
        assign_step_zero = tf.assign(global_step, 0)
        init_step = sess.run(assign_step_zero)
    if not initial_run:
        saver.restore(sess, model_to_use)
    for it in range(n_epochs):
        increment_global_step_op = tf.assign(global_step, global_step + 1)
        step = sess.run(increment_global_step_op)
        current_step = sess.run(global_step)
        train_loss = 0
        if (full_training):
            input_matrix = train_data[0:train_data.shape[0] // batch_size * batch_size, :]
            for lower in range(0, input_matrix.shape[0], batch_size):
                upper = min(lower + batch_size, input_matrix.shape[0])

                X_mb = input_matrix[lower:upper, :]
                _, D_loss_curr = sess.run(
                    [Solver, vae_loss], feed_dict={X: X_mb, time_step: current_step,
                                                           size: batch_size, is_training: True})
                train_loss += D_loss_curr
    os.makedirs(os.path.dirname(model_to_use), exist_ok=True)
    save_path = saver.save(sess, model_to_use)
    print("Model saved in file: %s" % save_path)
    print(f"total number of trained epochs is {current_step}")

def restore():
    saver.restore(sess, model_to_use)
def vector_batch_removal(inp, batch_key1, batch_key2):
    # projecting data to latent space
    latent_all = give_me_latent(inp.X)
    latent_ann = sc.AnnData(latent_all)
    latent_ann.obs["cell_type"] = inp.obs["Cell types"].tolist()
    latent_ann.obs["batch"] = inp.obs[batch_key1].tolist()
    latent_ann.obs[batch_key1] = inp.obs[batch_key1].tolist()
    latent_ann.obs[batch_key2] = inp.obs[batch_key2].tolist()
    unique_cell_types = np.unique(latent_ann.obs["cell_type"])
    not_shared_cell_types = []
    shared_anns = []
    not_shared_ann = []

    for cell_type in unique_cell_types:
        temp_cell = latent_ann[latent_ann.obs["cell_type"] == cell_type]
        if (len(np.unique(temp_cell.obs["batch"])) < 2):
            cell_type_ann = latent_ann[latent_ann.obs["cell_type"] == cell_type]
            not_shared_ann.append(cell_type_ann)
            continue
        print(f"{cell_type}")
        temp_cell = latent_ann[latent_ann.obs["cell_type"] == cell_type]
        batch_list = {}
        batch_ind = {}
        max_batch = 0
        max_batch_ind = ""
        batchs = np.unique(temp_cell.obs["batch"])
        for i in batchs:
            temp = temp_cell[temp_cell.obs["batch"] == i]
            temp_ind = temp_cell.obs["batch"] == i
            if max_batch < len(temp):
                max_batch = len(temp)
                max_batch_ind = i
            batch_list[i] = temp
            batch_ind[i] = temp_ind

        max_batch_ann = batch_list[max_batch_ind]
        detla_vecs = {}
        # Extract arrays and modify, avoiding view modification warnings
        for study in batch_list:
            delta = np.average(max_batch_ann.X, axis=0) - np.average(batch_list[study].X, axis=0)
            # Extract array, modify, and create new AnnData to avoid view issues
            modified_X = delta + batch_list[study].X
            batch_list[study] = sc.AnnData(modified_X, obs=batch_list[study].obs.copy(), var=batch_list[study].var.copy())
            # Copy temp_cell before modifying to avoid view issues
            if temp_cell.is_view:
                temp_cell = temp_cell.copy()
            temp_cell[batch_ind[study]].X = modified_X
        shared_anns.append(temp_cell)
    all_shared_ann = sc.AnnData.concatenate(*shared_anns)

    if (len(not_shared_cell_types) < 1):
        # reconstructing data to gene epxression space
        corrected = sc.AnnData(reconstruct(all_shared_ann.X, use_data=True))
        corrected.obs["cell_type"] = all_shared_ann.obs["cell_type"].tolist()
        corrected.obs[batch_key1] = all_shared_ann.obs[batch_key1].tolist()
        corrected.obs[batch_key2] = all_shared_ann.obs[batch_key2].tolist()

        corrected.var_names = inp.var_names.tolist()
        return corrected, all_shared_ann

    else:
        all_not_shared_ann = sc.AnnData.concatenate(*not_shared_ann)
        all_corrected_data = sc.AnnData.concatenate(all_shared_ann, all_not_shared_ann)
        # reconstructing data to gene epxression space
        corrected = sc.AnnData(reconstruct(all_corrected_data.X, use_data=True))
        corrected.obs["cell_type"] = all_shared_ann.obs["cell_type"].tolist() + all_not_shared_ann.obs[
            "cell_type"].tolist()
        corrected.obs["study"] = all_shared_ann.obs[batch_key1].tolist() + all_not_shared_ann.obs["batch"].tolist()
        corrected.var_names = data.var_names.tolist()
        # shared cell_types
        corrected_shared = sc.AnnData(reconstruct(all_shared_ann.X, use_data=True))
        corrected_shared.obs["cell_type"] = all_shared_ann.obs["cell_type"].tolist()
        corrected_shared.obs["study"] = all_shared_ann.obs[batch_key1].tolist()
        corrected_shared.var_names = inp.var_names.tolist()
        return corrected_shared, all_shared_ann


if __name__ == "__main__":
    # sc.pp.pca(data, svd_solver="arpack")
    # sc.pp.neighbors(data, n_neighbors=25)
    # sc.tl.umap(data)
    # import matplotlib
    # sc.pl.umap(data, legend_loc=False,  palette=matplotlib.rcParams["axes.prop_cycle"] , color=['Dataset'], save="orig_mouse_datasets.png", frameon=False, show=False)
    # sc.pl.umap(data, legend_loc=False, palette=matplotlib.rcParams["axes.prop_cycle"], color=['Cell types'], save="orig_mouse_Cell types.png", frameon=False, show=False)
    # sc.pl.umap(data, legend_loc=False, palette=matplotlib.rcParams["axes.prop_cycle"], color=['Organ groups'], save="orig_mouse_Organ groups.png", frameon=False, show=False)
    # X_pca = data.obsm["X_pca"]
    # labels = data.obs["Dataset"].tolist()
    # print(f"average silhouette_score for original mouse :{sk.metrics.silhouette_score(X_pca,labels,sample_size=57300, random_state=2)}")
    train(300)
    # restore()
    corrected_mouse_atlas, latent_batch = vector_batch_removal(data, "Dataset", "Organ groups")
    output_path = "../data/reconstructed/scGen/mouse_atlas.h5ad"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    corrected_mouse_atlas.write(output_path)
    # sc.pp.pca(corrected_mouse_atlas, svd_solver="arpack")
    # sc.pp.neighbors(corrected_mouse_atlas, n_neighbors=25)
    # sc.tl.umap(corrected_mouse_atlas)
    # sc.pl.umap(corrected_mouse_atlas, palette=matplotlib.rcParams["axes.prop_cycle"], color=['Dataset'], save="corrected_mouse_datasets_dsm.png",
    #            frameon=False, show=False)
    # sc.pl.umap(corrected_mouse_atlas, palette=matplotlib.rcParams["axes.prop_cycle"], color=['cell_type'], save="corrected_mouse_Cell types_dsm.png",
    #            frameon=False, show=False)
    # sc.pl.umap(corrected_mouse_atlas, palette=matplotlib.rcParams["axes.prop_cycle"], color=['Organ groups'], save="corrected_mouse_Organ groups_dsm.png",
    #            frameon=False, show=False)
    # X_pca = corrected_mouse_atlas.obsm["X_pca"]
    # labels2 = corrected_mouse_atlas.obs["Dataset"].tolist()
    # print(f"average silhouette_score for scGen :{sk.metrics.silhouette_score(X_pca,labels2,sample_size=57300, random_state=2)}")



