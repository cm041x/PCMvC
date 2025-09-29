import scipy.io as sio
import numpy as np
import scipy.sparse as ss
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering, KMeans
from scipy.optimize import linear_sum_assignment
from collections import Counter
import torch

from clusteringPerformance import clusteringMetrics
from datapre_canntlink import process_views



# -------------------- Data Loading --------------------
def load_data(path, dataset):
    """
    Load features and labels from .mat file.
    Normalize each view and convert sparse matrices to dense if necessary.
    """
    data = sio.loadmat(path + dataset + '.mat')
    features = data['X']
    feature_list = []
    labels = data['Y'].flatten()
    labels = labels - min(set(labels))  # ensure labels start from 0

    for i in range(features.shape[1]):
        features[0][i] = normalize(features[0][i])
        feature = features[0][i]
        if ss.isspmatrix_csr(feature):
            feature = feature.todense()
        feature_list.append(feature)
    return feature_list, labels




# -------------------- KNN Adjacency Construction --------------------
def construct_knn_adjacency(features, k=5):
    """
    Construct K-nearest neighbor adjacency matrices for each view.
    """
    adj_matrices = []
    for feature in features:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(feature)
        distances, indices = nbrs.kneighbors(feature)

        adj = np.zeros((feature.shape[0], feature.shape[0]))
        for i in range(indices.shape[0]):
            for j in indices[i]:
                adj[i, j] = 1
                adj[j, i] = 1  # make it symmetric
        adj_matrices.append(torch.tensor(adj, dtype=torch.float32))
    return adj_matrices


# -------------------- Dempster-Shafer Combination --------------------
def dempster_combination_thre(bpas, threshold_factors):
    """
    Fuse adjacency matrices (BPAs) using Dempster's rule
    with view-specific threshold factors.
    """
    combined_bpa = torch.zeros_like(bpas[0])
    for i, bpa1 in enumerate(bpas):
        for j, bpa2 in enumerate(bpas):
            if i != j:
                combined_bpa += bpa1 * bpa2 * threshold_factors[i] * threshold_factors[j]
    normalization_factor = combined_bpa.sum()
    if normalization_factor > 0:
        combined_bpa /= normalization_factor
    return combined_bpa


def calculate_threshold_factors(bpas):
    """
    Calculate threshold factors based on the entropy of each BPA.
    Lower entropy → higher confidence (larger threshold factor).
    """
    threshold_factors = []
    for bpa in bpas:
        entropy = -torch.sum(bpa * torch.log(bpa + 1e-10))
        informational_content = 1 / (1 + entropy)
        threshold_factors.append(informational_content)
    return threshold_factors


# -------------------- Constraint Handling --------------------
def apply_cannot_link_constraints(adj_matrix, cannot_link_matrix):
    """
    Enforce cannot-link constraints by setting corresponding edges to 0.
    """
    return np.where(cannot_link_matrix == 1, 0, adj_matrix)


# -------------------- Clustering and Metrics --------------------
def spectral_cluster(adj_matrix, n_clusters=3):
    """
    Perform spectral clustering on the fused adjacency matrix.
    """
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    return clustering.fit_predict(adj_matrix)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using the Hungarian algorithm.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(-1, 1), u[1].reshape(-1, 1)], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity_score(clusters, labels):
    """
    Compute clustering purity.
    """
    total_samples = len(labels)
    correct = 0
    for cluster in set(clusters):
        cluster_labels = [labels[i] for i in range(len(labels)) if clusters[i] == cluster]
        most_common_label = Counter(cluster_labels).most_common(1)[0][1]
        correct += most_common_label
    return correct / total_samples


# -------------------- Random Seed --------------------
def setup_seed(seed):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ---- Linear Feature Filtering with Loss Tracking ----
def initialize_Uv(Xv, c, random_state=42):
    kmeans = KMeans(n_clusters=c, random_state=random_state).fit(Xv.T)
    Uv = kmeans.cluster_centers_.T
    return Uv

def initialize_Vv(Xv, Uv):
    Vv = np.dot(np.linalg.pinv(Uv.T @ Uv), Uv.T @ Xv)
    return Vv

def frobenius_loss(Xv_list, Uv_list, Vv_list):
    total_loss = 0.0
    for Xv, Uv, Vv in zip(Xv_list, Uv_list, Vv_list):
        total_loss += np.linalg.norm(Xv - np.dot(Uv, Vv), 'fro') ** 2
    return total_loss

def update_Uv_gd(Xv, Uv, Vv, learning_rate=0.01):
    gradient = -2 * np.dot((Xv - np.dot(Uv, Vv)), Vv.T)
    Uv -= learning_rate * gradient
    Uv = np.maximum(Uv, 0)
    return Uv

def update_Vv_gd(Xv, Uv, Vv, learning_rate=0.01):
    Vv = np.maximum(Vv, 0)
    gradient = -2 * np.dot(Uv.T, (Xv - np.dot(Uv, Vv)))
    Vv -= learning_rate * gradient
    U, _, Vt = np.linalg.svd(Vv, full_matrices=False)
    Vv = np.dot(U, Vt)
    return Vv

def optimize_multiview(Xv_list, c, iterations=200, learning_rate=0.0001, tol=1e-6, patience=50, log_every=10):
    Uv_list = [initialize_Uv(Xv, c) for Xv in Xv_list]
    Vv_list = [initialize_Vv(Xv, Uv) for Xv, Uv in zip(Xv_list, Uv_list)]

    prev_loss = frobenius_loss(Xv_list, Uv_list, Vv_list)
    no_improve = 0
    losses = [prev_loss]

    for it in range(iterations):
        for i in range(len(Xv_list)):
            Xv = Xv_list[i]
            Uv_list[i] = update_Uv_gd(Xv, Uv_list[i], Vv_list[i], learning_rate)
            Vv_list[i] = update_Vv_gd(Xv, Uv_list[i], Vv_list[i], learning_rate)

        cur_loss = frobenius_loss(Xv_list, Uv_list, Vv_list)
        losses.append(cur_loss)
        if (it + 1) % log_every == 0:
            print(f"[Iter {it + 1:4d}] Frobenius loss = {cur_loss:.6f}")

        if cur_loss < prev_loss:
            improve = prev_loss - cur_loss
            if improve < tol:
                no_improve += 1
            else:
                no_improve = 0
        else:
            no_improve = 0
        prev_loss = cur_loss

        if patience is not None and patience > 0 and no_improve >= patience:
            print(f"Early stopping at iter {it+1} (no_improve={no_improve}).")
            break

    return Uv_list, Vv_list


# -------------------- Main Pipeline --------------------
def main(path, data, max_iter, tol, learn, k, p):
    """
    Complete pipeline:
    - Load data
    - Learn view-specific representations
    - Construct adjacency matrices
    - Fuse views with Dempster-Shafer rule
    - Apply cannot-link constraints
    - Perform spectral clustering
    - Evaluate metrics
    """
    features, labels = load_data(path, data)
    n_class = len(np.unique(labels))
    setup_seed(42)

    new_features = [f.T for f in features]
    U_list, V_list = optimize_multiview(new_features, n_class,
                                        iterations=max_iter, learning_rate=learn,
                                        tol=tol)

    V_list = [V.T for V in V_list]
    new_adj_matrices = construct_knn_adjacency(V_list, k)
    threshold_factors = calculate_threshold_factors(new_adj_matrices)
    new_combined_adj = dempster_combination_thre(new_adj_matrices, threshold_factors)
    cannot_link_matrix = process_views(V_list, p)
    updated_adj_matrix = apply_cannot_link_constraints(new_combined_adj, cannot_link_matrix)

    pred_labels = spectral_cluster(updated_adj_matrix, n_class)
    acc = cluster_acc(labels, pred_labels)

    NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetrics(labels, pred_labels)
    return acc, NMI, Purity, ARI, Fscore, Precision, Recall


# -------------------- Run Example --------------------
if __name__ == '__main__':
    path = './data/'
    dataset_dict = {1: 'BBCSports',2: 'esp_game', 3: 'NottingHill'}
    dataset_params = {
        1: {'k': 108, 'p': 50, 'learn': 0.006, 'iterations': 200},
        2: {'k': 1576, 'p': 60, 'learn': 0.02904993490601889, 'iterations': 200},
        3: {'k': 932, 'p': 50, 'learn': 0.07897219752341908, 'iterations': 200},
    }

    for idx in [1, 2]:
        save_path = f'result.txt'
        for tol in [1e-5]:
            params = dataset_params[idx]
            acc, NMI, Purity, ARI, Fscore, Precision, Recall = main(
                path, dataset_dict[idx],
                max_iter=params['iterations'], tol=tol,
                learn=params['learn'], k=params['k'], p=params['p']
            )
            with open(save_path, "a") as f:
                f.write(f"\ndata: {dataset_dict[idx]},\n")
                f.write(f"\tacc: {acc * 100:.2f}\n")
                f.write(f"\tnmi: {NMI * 100:.2f}\n")
                f.write(f"\tari: {ARI * 100:.2f}\n")
                f.write(f"\tfscore: {Fscore * 100:.2f}\n")
