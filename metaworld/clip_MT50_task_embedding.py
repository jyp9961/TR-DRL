import clip
import numpy as np
import torch
import utils
from MT50_task_descriptions import MT50_task_descriptions
from sklearn import decomposition

if __name__ == '__main__':
    print('MT50_task_descriptions:', MT50_task_descriptions)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(name="RN50", device=device, download_root='./')

    text = clip.tokenize(list(MT50_task_descriptions.values())).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
    
    text_features_numpy = text_features.detach().cpu().numpy()
    print(text_features_numpy)

    pca = decomposition.PCA(n_components = 50)
    pca.fit(text_features_numpy)
    X = pca.transform(text_features_numpy)
    print(X, np.shape(X), np.max(X), np.min(X))

    MT50_task_embedding = {} 
    for i in range(50):
        MT50_task_embedding[list(MT50_task_descriptions.keys())[i]] = X[i]
    print(MT50_task_embedding)
    np.save('MT50_task_embedding.npy', MT50_task_embedding)

    test_embedding = np.load('MT50_task_embedding.npy', allow_pickle=True).item()
    print((test_embedding['door-open'] == MT50_task_embedding['door-open']).all())