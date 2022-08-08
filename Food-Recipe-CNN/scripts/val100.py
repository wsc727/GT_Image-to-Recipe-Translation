# This scripts randomly picks 100 recipe images out of the val dataset
# for comparing the two models.
import pickle
import numpy as np
import lmdb

with open('data/val_keys.pkl','rb') as f:
    recipe_ids = pickle.load(f)

recipe_db = lmdb.open('data/val_lmdb', max_readers=1,
                      readonly=True, lock=False, readahead=False, meminit=False)

image_ids = []
while len(image_ids) < 100:
    recipe_idx = np.random.choice(range(len(recipe_ids)))
    recipe_id = recipe_ids[recipe_idx]
    with recipe_db.begin(write=False) as txn:
        recipe = txn.get(recipe_id.encode('latin1'))
    imgs = pickle.loads(recipe, encoding='latin1')['imgs']
    if len(imgs) > 0:
        image_ids.append(imgs[0]['id'])

print(image_ids)

with open('data/val100_images.pkl', 'wb') as f:
    pickle.dump(image_ids, f)
