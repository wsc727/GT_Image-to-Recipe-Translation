# Food-Recipe-CNN

## Setup
```bash
conda env create -f environment.yaml
conda activate food-recipe-cnn
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
```

## 1. nearest\_neighbors.py
This trains an index of Approximate Nearest Neighbors. It's used for predict.py.

```bash
python nearest_neighbors.py test
```

### Input
* `data/recipe1M/layer2.json`: id, image urls
* `data/images/test`: Recipe images, test

### Output
* `data/test_ann_recipe_ids.pkl`: Recipe ids indexed by Approximate Nearest Neighbors. Not unique.
* `data/test_ann_image_ids.pkl`: Image ids indexed by Approximate Nearest Neighbors.
* `data/test_ann_ipca.pkl`: IncrementalPCA model for Approximate Nearest Neighbors.
* `data/test_ann_nmslib.bin`: NMSLIB index for Approximate Nearest Neighbors.

## 2. topic\_modeling.py
This generates 300 categories generated from the train dataset.
Then all recipes ids in train, val, and test datasets will get a top category prediction.

```bash
python topic_modeling.py nmf 30
python topic_modeling.py nmf 300
python topic_modeling.py w2v 30
python topic_modeling.py w2v 300
```

### Input
* `data/recipe1M/layer1.json`: ingredients, url, partition, title, id, instructions

### Output
* `data/(nmf|w2v)_(30|300)_categories.txt`: 30 or 300 lines of categories generated primarily for train
* `data/(nmf|w2v)_(30|300)_categories_lmdb`: { recipe\_id => (30|300)\_categories\_idx } mapping for train, val, and test

## 3. train.py
This fine-tunes the last reshaped layers of InceptionV3 using the predicted top categories for the train dataset,
and validates it using ones for the val dataset.

```bash
python train.py nmf 30
python train.py nmf 300
python train.py w2v 30
python train.py w2v 300
```

### Input
* `data/(nmf|w2v)_(30|300)_categories_lmdb`: { recipe\_id => (30|300)\_categories\_idx } mapping for train, val, and test
* `data/train_keys.pkl`: List of recipe ids, train
* `data/train_lmdb`: Recipe id to ingrs, intrs, classes, and imgs mapping, train
* `data/val_keys.pkl`: List of recipe ids, val
* `data/val_lmdb`: Recipe id to ingrs, intrs, classes, and imgs mapping, val
* `data/images/train`: Recipe images, train
* `data/images/val`: Recipe images, val

### Output
* `snapshots/(nmf|w2v)_(30|300)_model_eXXX_v-Y.YYY.pth.tar`: X=epoch, Y=best\_acc

## 4. predict.py
Use a trained model to predict a recipe.

```bash
python predict.py nmf 30  snapshots/nmf_model_eXXX_v-Y.YYY.pth.tar [image].jpg
python predict.py nmf 300 snapshots/nmf_model_eXXX_v-Y.YYY.pth.tar [image].jpg
python predict.py w2v 30  snapshots/w2v_model_eXXX_v-Y.YYY.pth.tar [image].jpg
python predict.py w2v 300 snapshots/w2v_model_eXXX_v-Y.YYY.pth.tar [image].jpg
```

### Input
* `data/recipe1M/layer1.json`: ingredients, url, partition, title, id, instructions
* `data/recipe1M/layer2.json`: id, image urls
* `data/(nmf|w2v)_(30|300)_categories.txt`: 30 or 300 lines of categories generated primarily for train
* `data/(nmf|w2v)_(30|300)_categories_lmdb`: { recipe\_id => (30|300)\_categories\_idx } mapping for train, val, and test
* `data/images/test`: Recipe images, test
* `data/test_ann_recipe_ids.pkl`: Recipe ids indexed by Approximate Nearest Neighbors. Not unique.
* `data/test_ann_image_ids.pkl`: Image ids indexed by Approximate Nearest Neighbors.
* `data/test_ann_ipca.pkl`: IncrementalPCA model for Approximate Nearest Neighbors.
* `data/test_ann_nmslib.bin`: NMSLIB index for Approximate Nearest Neighbors.
* `snapshots/(nmf|w2v)_model_eXXX_v-Y.YYY.pth.tar`: X=epoch, Y=best\_acc
