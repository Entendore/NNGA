import numpy as np
import json
import os
import hnswlib
from typing import List, Tuple, Optional, Dict

class VectorDatabase:
    def __init__(self, dim: int, ivf_clusters: int = 0, use_hnsw: bool = False, ef_construction: int = 200, M: int = 16):
        """
        Initialize the vector database with optional IVF or HNSW index.
        
        Args:
            dim (int): Vector dimension.
            ivf_clusters (int): Number of IVF clusters (0 to disable).
            use_hnsw (bool): Whether to use HNSW index.
            ef_construction (int): HNSW construction parameter.
            M (int): HNSW construction parameter.
        """
        self.dim = dim
        self.vectors: Dict[str, np.ndarray] = {}

        # IVF parameters
        self.ivf_clusters = ivf_clusters
        self.ivf_index: Dict[int, List[str]] = {}
        self.centroids: Optional[np.ndarray] = None

        # HNSW parameters
        self.use_hnsw = use_hnsw
        self.hnsw_index = None
        self.hnsw_label_to_id: Dict[int, str] = {}
        self.hnsw_id_to_label: Dict[str, int] = {}
        self.ef_construction = ef_construction
        self.M = M
        self.next_hnsw_label = 0  # internal integer label for hnswlib

    def add_vector(self, id: str, vector: List[float]) -> None:
        v = np.array(vector, dtype=np.float32)
        if v.shape[0] != self.dim:
            raise ValueError(f"Vector dimension must be {self.dim}")
        self.vectors[id] = v

        if self.use_hnsw:
            if self.hnsw_index is None:
                self._init_hnsw_index(max_elements=10000)  # start with 10k capacity, auto-resize supported
            if id not in self.hnsw_id_to_label:
                label = self.next_hnsw_label
                self.hnsw_label_to_id[label] = id
                self.hnsw_id_to_label[id] = label
                self.hnsw_index.add_items(v.reshape(1, -1), np.array([label]))
                self.next_hnsw_label += 1

        if self.ivf_clusters > 0:
            self._assign_to_cluster(id, v)

    def _init_hnsw_index(self, max_elements: int):
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.dim)
        self.hnsw_index.init_index(max_elements=max_elements, ef_construction=self.ef_construction, M=self.M)
        self.hnsw_index.set_ef(50)  # ef parameter for query time (higher = better accuracy, slower)

    def add_vectors_batch(self, id_vec_pairs: List[Tuple[str, List[float]]]) -> None:
        for id, vec in id_vec_pairs:
            self.add_vector(id, vec)

    def remove_vector(self, id: str) -> None:
        if id in self.vectors:
            del self.vectors[id]
        if self.use_hnsw and id in self.hnsw_id_to_label:
            label = self.hnsw_id_to_label[id]
            self.hnsw_index.mark_deleted(label)
            del self.hnsw_label_to_id[label]
            del self.hnsw_id_to_label[id]
        if self.ivf_clusters > 0:
            for cluster_id, ids in self.ivf_index.items():
                if id in ids:
                    ids.remove(id)

    def update_vector(self, id: str, vector: List[float]) -> None:
        self.remove_vector(id)
        self.add_vector(id, vector)

    def get_vector(self, id: str) -> Optional[np.ndarray]:
        return self.vectors.get(id, None)

    def build_ivf_index(self):
        if len(self.vectors) == 0:
            raise RuntimeError("No vectors to index")
        if self.ivf_clusters <= 0:
            raise RuntimeError("IVF clusters must be > 0 to build index")

        all_vecs = np.stack(list(self.vectors.values()))
        from sklearn.cluster import KMeans  # delayed import to avoid heavy dependency if unused
        kmeans = KMeans(n_clusters=self.ivf_clusters, random_state=42)
        labels = kmeans.fit_predict(all_vecs)
        self.centroids = kmeans.cluster_centers_
        self.ivf_index = {i: [] for i in range(self.ivf_clusters)}

        for idx, id in enumerate(self.vectors.keys()):
            cluster_id = labels[idx]
            self.ivf_index[cluster_id].append(id)

    def _assign_to_cluster(self, id: str, vec: np.ndarray) -> None:
        if self.centroids is None:
            return
        dists = np.linalg.norm(self.centroids - vec, axis=1)
        cluster_id = int(np.argmin(dists))
        if cluster_id not in self.ivf_index:
            self.ivf_index[cluster_id] = []
        if id not in self.ivf_index[cluster_id]:
            self.ivf_index[cluster_id].append(id)

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

    def _euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.linalg.norm(v1 - v2)

    def search(self,
               query_vector: List[float],
               top_k: int = 5,
               metric: str = "cosine",
               ivf_probe: int = 3) -> List[Tuple[str, float]]:
        qv = np.array(query_vector, dtype=np.float32)
        if qv.shape[0] != self.dim:
            raise ValueError(f"Query vector dimension must be {self.dim}")

        if metric not in ("cosine", "euclidean"):
            raise ValueError("Metric must be 'cosine' or 'euclidean'")

        if self.use_hnsw and self.hnsw_index is not None:
            # hnswlib only supports cosine or L2. Here we only do cosine for simplicity.
            labels, distances = self.hnsw_index.knn_query(qv.reshape(1, -1), k=top_k)
            results = []
            for label, dist in zip(labels[0], distances[0]):
                if label == -1:
                    continue
                id = self.hnsw_label_to_id.get(label, None)
                if id is not None:
                    if metric == "cosine":
                        # hnswlib distance for cosine is 1 - cosine similarity
                        score = 1 - dist
                    else:
                        score = dist  # L2 distance
                    results.append((id, score))
            return results

        # IVF or brute-force search fallback
        search_ids = self.vectors.keys()
        if self.ivf_clusters > 0 and self.centroids is not None:
            dists_to_centroids = np.linalg.norm(self.centroids - qv, axis=1)
            probe_clusters = np.argsort(dists_to_centroids)[:ivf_probe]
            candidate_ids = []
            for c in probe_clusters:
                candidate_ids.extend(self.ivf_index.get(c, []))
            search_ids = candidate_ids

        results = []
        for id in search_ids:
            vec = self.vectors[id]
            if metric == "cosine":
                score = self._cosine_similarity(qv, vec)
            else:
                score = self._euclidean_distance(qv, vec)
            results.append((id, score))

        if metric == "cosine":
            results.sort(key=lambda x: x[1], reverse=True)
        else:
            results.sort(key=lambda x: x[1])
        return results[:top_k]

    def save(self, folder_path: str) -> None:
        os.makedirs(folder_path, exist_ok=True)
        ids = list(self.vectors.keys())
        vecs = np.stack([self.vectors[id] for id in ids])
        np.save(os.path.join(folder_path, "vectors.npy"), vecs)
        with open(os.path.join(folder_path, "ids.json"), "w") as f:
            json.dump(ids, f)

        if self.ivf_clusters > 0 and self.centroids is not None:
            with open(os.path.join(folder_path, "ivf_index.json"), "w") as f:
                json.dump(self.ivf_index, f)
            np.save(os.path.join(folder_path, "centroids.npy"), self.centroids)

        if self.use_hnsw and self.hnsw_index is not None:
            self.hnsw_index.save_index(os.path.join(folder_path, "hnsw_index.bin"))
            # Save mapping label <-> id
            with open(os.path.join(folder_path, "hnsw_label_to_id.json"), "w") as f:
                json.dump(self.hnsw_label_to_id, f)
            with open(os.path.join(folder_path, "hnsw_id_to_label.json"), "w") as f:
                json.dump(self.hnsw_id_to_label, f)

    def load(self, folder_path: str) -> None:
        vectors_path = os.path.join(folder_path, "vectors.npy")
        ids_path = os.path.join(folder_path, "ids.json")
        ivf_index_path = os.path.join(folder_path, "ivf_index.json")
        centroids_path = os.path.join(folder_path, "centroids.npy")

        if not os.path.exists(vectors_path) or not os.path.exists(ids_path):
            raise FileNotFoundError("Vector files missing")

        vecs = np.load(vectors_path)
        with open(ids_path, "r") as f:
            ids = json.load(f)

        self.vectors = {id: vec for id, vec in zip(ids, vecs)}

        if os.path.exists(ivf_index_path) and os.path.exists(centroids_path):
            with open(ivf_index_path, "r") as f:
                self.ivf_index = json.load(f)
            self.centroids = np.load(centroids_path)
            self.ivf_clusters = len(self.centroids)
        else:
            self.ivf_index = {}
            self.centroids = None
            self.ivf_clusters = 0

        hnsw_index_path = os.path.join(folder_path, "hnsw_index.bin")
        label_to_id_path = os.path.join(folder_path, "hnsw_label_to_id.json")
        id_to_label_path = os.path.join(folder_path, "hnsw_id_to_label.json")

        if self.use_hnsw and os.path.exists(hnsw_index_path):
            self._init_hnsw_index(max_elements=len(self.vectors) + 1000)
            self.hnsw_index.load_index(hnsw_index_path)
            with open(label_to_id_path, "r") as f:
                self.hnsw_label_to_id = json.load(f)
            with open(id_to_label_path, "r") as f:
                self.hnsw_id_to_label = json.load(f)
            self.next_hnsw_label = max(map(int, self.hnsw_label_to_id.keys())) + 1
        else:
            self.hnsw_index = None
            self.hnsw_label_to_id = {}
            self.hnsw_id_to_label = {}
            self.next_hnsw_label = 0

    def stats(self) -> dict:
        return {
            "num_vectors": len(self.vectors),
            "dimension": self.dim,
            "ivf_clusters": self.ivf_clusters,
            "use_hnsw": self.use_hnsw,
            "hnsw_index_size": self.hnsw_index.get_current_count() if self.hnsw_index else 0,
        }
