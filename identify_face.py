from deepface import DeepFace

if __name__ == "__main__":
    dfs = DeepFace.find(img_path="../../MyFace Dataset/probe/106.png", db_path="../../MyFace Dataset/gallery",
                        detector_backend="retinaface", model_name="ArcFace", distance_metric="euclidean_l2")
    print(dfs)
