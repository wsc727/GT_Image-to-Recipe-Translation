# This script tries to look as similar to the original deploy/core_algo_tests.ipynb as possible.
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from train import initialize_incv3
from nearest_neighbors import initialize_vgg16, get_image_vgg
import nmslib
import numpy as np
import simplejson as json
import subprocess
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from nearest_neighbors import init_ann_index
import pickle
import lmdb
import os

K = 25

# Fetch the top category name of recipe_id
def get_category(recipe_id):
    with categories_db.begin(write=False) as txn:
        label = int(txn.get(recipe_id.encode('latin1')))
    if label == -1:
        return 'no_cat'
    else:
        return categories[label]

def load_ipca():
    with open('data/test_ann_ipca.pkl', 'rb') as f:
        ipca = pickle.load(f)
    return ipca

# Get the fine-tuned InceptionV3
def load_incv3(model_path, num_classes=300):
    state = torch.load(model_path)
    model_inc = initialize_incv3(num_classes)
    model_inc.load_state_dict(state['state_dict'])
    return model_inc

def load_models(model_path, num_classes=300):
    global model_inc, feat_extractor
    model_inc = load_incv3(model_path, num_classes)
    model_inc.eval()
    feat_extractor = initialize_vgg16()
    feat_extractor.eval()

def get_image_inc(path):
    img = Image.open(path).convert('RGB')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize,
    ])
    return transform(img)

def get_closest_images_nmslib(query_features, num_results=K):
    return index_ann.knnQuery(query_features, k = num_results)

def weighting_neural_net_inputs(query_features, probabilities):
    """Combine outputs from Inceptionv3 and VGG-16 to a result list.

    Argument:
    query_features: query image fingerprint from VGG-16
    probabilities: Inception's category probabilities

    Return: final list containing category, inception's confidence,
            recipe id, image index and image path.
    """
    # do a query on image
    idx_closest, distances = get_closest_images_nmslib(query_features)

    # Don't forget to adjust string slicing for second hdf5
    # Labels only from ANN
    predicted_labels = [get_category(recipe_ids[i]) for i in idx_closest] # dummy. TODO: create this in test.py and replace it

    # Results only from ANN
    predicted_ids = [[image_ids[i]] for i in idx_closest]

    pred_categories = []

    for i, x in enumerate(np.argsort(-probabilities)[:K]):
        confidence = -np.sort(-probabilities)[i]
        # print(categories[x], confidence)
        pred_categories.append([categories[x], confidence])

    predicted_labels_with_weights = []
    for iii in predicted_labels:
        for iiii, ii in enumerate(pred_categories):
            no_result = False
            if ii[0] == iii:
                predicted_labels_with_weights.append([iii, ii[1]])
                break
            if iiii == len(pred_categories)-1:
                predicted_labels_with_weights.append([iii, 0])

    predicted_labels_with_meta = [xi+yi for xi, yi in zip(predicted_labels_with_weights, predicted_ids)]
    final_result = sorted(predicted_labels_with_meta, key=lambda predicted_labels_with_meta: predicted_labels_with_meta[1], reverse=True)

    return final_result

def model_predict(query_img_path):
    x = get_image_vgg(query_img_path).unsqueeze(0) # the model expects a batch of images, thus unsqueeze
    query_features = feat_extractor.forward(x).detach().numpy()

    # project it into pca space
    pca_query_features = ipca.transform(query_features)[0]

    x = get_image_inc(query_img_path).unsqueeze(0) # Preprocess query image for Inception
    probabilities = model_inc.forward(x)[0].detach().numpy() # Get Inception's category probabilities not sorted

    final_result = weighting_neural_net_inputs(pca_query_features, probabilities) # Get final food result
    return final_result

# Load categories
topic_modeling = sys.argv[1]
n_topics = int(sys.argv[2])
with open(f"data/{topic_modeling}_{n_topics}_categories.txt") as f:
    categories = f.read().split('\n')
categories_db = lmdb.open(f"data/{topic_modeling}_{n_topics}_categories_lmdb", max_readers=1,
                          readonly=True, lock=False, readahead=False, meminit=False)

# Load image/recipe ids indexed for Approximate Nearest Neighbors
with open('data/test_ann_image_ids.pkl', 'rb') as f:
    image_ids = pickle.load(f)
with open('data/test_ann_recipe_ids.pkl', 'rb') as f:
    recipe_ids = pickle.load(f)

# Load fine-turned InceptionV3 and pre-trained VGG16
load_models(sys.argv[3], n_topics)

# Load an IncrementalPCA model
ipca = load_ipca()

# Load an Approximate Nearest Neighbor model
index_ann = init_ann_index('data/test_ann_nmslib.bin')

def main(im_path):
    # Predict K recipes ordered by similarity
    result_list = model_predict(im_path)

    # just show only the most similar recipe for now
    image_id = result_list[0][2]

    # Translate image id to recipe id using layer2
    recipe_id = json.loads(
        subprocess.check_output(['bash', '-c', f'cat data/recipe1M/layer2.json | grep "{image_id}"']).rstrip()[0:-1] # strip ",\n"
    )['id']

    # Find the recipe from layer1
    recipe = subprocess.check_output(['bash', '-c', f'cat data/recipe1M/layer1.json | grep "{recipe_id}"']).rstrip()[0:-1] # strip ",\n"
    recipe = json.loads(recipe)

    # Make it easier to read
    recipe['ingredients'] = [i['text'] for i in recipe['ingredients']]
    recipe['instructions'] = [i['text'] for i in recipe['instructions']]

    # Dump
    # print(json.dumps(recipe, indent='  '))
    print(f"{im_path}: [{recipe['title']}]({recipe['url']})")

# main(sys.argv[4])
with open('../Food-Recipe-CNN/data/val100_images.pkl','rb') as f:
    val_image_ids = pickle.load(f)
for image_id in val_image_ids:
    loader_path = os.path.join(*[image_id[i] for i in range(4)])
    path = f"data/images/val/{loader_path}/{image_id}"
    main(path)
