
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler, normalize
from Levenshtein import distance
import pickle as pkl
from abstract import MASAbstraction, AbstractedSubtask

file_path = 'workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa-diamond/abstracted_subtasks.pkl'

with open(file_path, 'rb') as f:
    subtasks = pkl.load(f)

for idx, subtask in enumerate(subtasks):
    print(f"\n=========== Subtask {idx} =============\nGenerate subtask: ", subtask.abstracted_objective)
    print("Inner objectives")
    for objective in subtask.objective:
        print(objective)
