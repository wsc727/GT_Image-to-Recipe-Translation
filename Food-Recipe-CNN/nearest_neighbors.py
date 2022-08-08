from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import os
import pickle
import simplejson as json
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import nmslib

class ImageLoader(Dataset):
    def __init__(self, partition, image_ids):
        self.partition = partition
        self.image_ids = image_ids

    def __getitem__(self, index):
        # Load the image
        loader_path = os.path.join(*[self.image_ids[index][i] for i in range(4)])
        path = f"data/images/{self.partition}/{loader_path}/{self.image_ids[index]}"
        return get_image_vgg(path)

    def __len__(self):
        return len(self.image_ids)

# Index parameters
M = 15
efC = 100
num_threads = 4
index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}
space_name = 'l2'
efS = 100
query_time_params = {'efSearch': efS}

def create_ann_index(features_ipca):
    # Intitialize the library, specify the space, the type of the vector and add data points
    index_ann = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
    index_ann.addDataPointBatch(features_ipca)
    index_ann.createIndex(index_time_params, print_progress=True)
    return index_ann

def init_ann_index(bin_path):
    # Intitialize the library, specify the space, the type of the vector and add data points 
    index_ann = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
    # Re-load the index and re-run queries
    index_ann.loadIndex(bin_path)
    # Setting query-time parameters and querying
    index_ann.setQueryTimeParams(query_time_params)
    return index_ann

def initialize_vgg16():
    feat_extractor = models.vgg16(pretrained=True)
    # Remove the last FC layer to get 4096 internal features instead of classification results
    # See also: https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html#vgg16
    feat_extractor.classifier = nn.Sequential(*list(feat_extractor.classifier.children())[:-1])
    return feat_extractor

# Get an image vector for VGG
def get_image_vgg(path):
    img = Image.open(path).convert('RGB')
    # VGG-16 expects a 224x224 image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return transform(img)

def main(partition):
    print('Loading layer2.json.')
    with open('data/recipe1M/layer2.json') as f:
        layer2 = json.load(f)

    print('Saving image/recipe id indexes.')
    recipe_ids = []
    image_ids = []
    for recipe in layer2:
        for image in recipe['images']:
            loader_path = os.path.join(*[image['id'][i] for i in range(4)])
            path = f"data/images/{partition}/{loader_path}/{image['id']}"
            if os.path.exists(path):
                recipe_ids.append(recipe['id'])
                image_ids.append(image['id'])
    with open(f"data/{partition}_ann_recipe_ids.pkl", 'wb') as f:
        pickle.dump(recipe_ids, f)
    with open(f"data/{partition}_ann_image_ids.pkl", 'wb') as f:
        pickle.dump(image_ids, f)

    print('Extracting features with pre-trained VGG-16.')
    feat_extractor = initialize_vgg16()
    feat_extractor.to(device)
    feat_extractor.eval()
    data_loader = DataLoader(
        dataset=ImageLoader(partition=partition, image_ids=image_ids),
        batch_size=25, num_workers=8, shuffle=False, pin_memory=True,
    )
    ipca = IncrementalPCA(n_components=512, batch_size=1024)
    features_all = None
    for images in tqdm(data_loader):
        features = feat_extractor.forward(images.to(device)).detach().cpu().numpy()
        features_all = features if features_all is None else np.concatenate([features_all, features])

    print('Training IncrementalPCA.')
    ipca.partial_fit(features_all)
    with open(f"data/{partition}_ann_ipca.pkl", 'wb') as f:
        pickle.dump(ipca, f)

    print('Creating an index of NMSLIB.')
    features_ipca = ipca.transform(features_all)
    index_ann = create_ann_index(features_ipca)
    index_ann.saveIndex(f"data/{partition}_ann_nmslib.bin")

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(sys.argv[1])
