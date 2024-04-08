import cv2
from sklearn.cluster import KMeans


def get_dominant_colors(array_image, clusters=3):
    frame = cv2.cvtColor(array_image, cv2.COLOR_BGR2RGB)
    frame = frame.reshape((frame.shape[0] * frame.shape[1], 3))
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(frame)
    return kmeans.cluster_centers_.astype(int)
