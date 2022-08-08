from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pandas as pd
import re
import simplejson as json
import sys
import numpy as np
import shutil
import os
import lmdb
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
import gensim
import copy
from sklearn.cluster import KMeans

def sentences_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True)) # deacc=True removes punctuations

def remove_stopwords(texts):
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'with', 'and', 'recipe', 'in', 'a', 's'])

    return [[word.strip() for word in simple_preprocess(str(doc))
             if word.strip() not in stop_words] for doc in texts]

def put_top_category_for_recipes(lmdb_name, df_title, w_vector):
    db = lmdb.open(lmdb_name, map_size=int(1e11))
    with db.begin(write=True) as txn:
        for i in tqdm(range(len(df_title))):
            one_hot_matrix = w_vector[i,:]
            if one_hot_matrix.any():
                category = np.argsort(one_hot_matrix)[::-1][0] # index of topics_category_300
            else: # all zeros
                category = -1 # no category
            recipe_id = df_title['id'][i]
            txn.put(recipe_id.encode('latin1'), str(category).encode('latin1'))

def create_clean_titled_dfs():
    print('Loading layer1.json.')
    with open('data/recipe1M/layer1.json') as f:
        layer1 = json.load(f)

    print('Generating title_clean.')
    dfs = []
    for partition in ['train', 'val', 'test']:
        layer1_data_list = []
        for i in range(len(layer1)):
            if layer1[i]['partition'] == partition:
                layer1_data_list.append({
                    'id': layer1[i]['id'],
                    'title_clean': re.sub(r'[^\w]', ' ', layer1[i]['title']).lower(),
                })
        dfs.append(pd.DataFrame(layer1_data_list))
    return dfs

def build_nmf(n_topics):
    df_train, df_val, df_test = create_clean_titled_dfs()
    words_train = remove_stopwords(list(sentences_to_words(df_train['title_clean'].values.tolist())))
    words_val   = remove_stopwords(list(sentences_to_words(df_val['title_clean'].values.tolist())))
    words_test  = remove_stopwords(list(sentences_to_words(df_test['title_clean'].values.tolist())))

    print('Training TfidVectorizer.')
    tfidf_vectorizer = TfidfVectorizer(max_df=0.99, max_features=None)
    tfidf = tfidf_vectorizer.fit_transform([' '.join(item) for item in words_train])

    print('Training NMF.')
    nmf_300 = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5, verbose=2, max_iter=5).fit(tfidf)

    print('Saving a list of categories.')
    nmf_embedding = nmf_300.transform(tfidf)
    top_idx = np.argsort(nmf_embedding, axis=0)[-1:]
    topics_category_300 = [' '.join(words_train[i]) for i in top_idx[0]]
    with open(f"data/nmf_{n_topics}_categories.txt", 'w') as f:
        f.write('\n'.join(topics_category_300))

    print('Predicting top categories for train.')
    lmdb_name = f"data/nmf_{n_topics}_categories_lmdb"
    if os.path.isdir(lmdb_name):
        shutil.rmtree(lmdb_name)
    put_top_category_for_recipes(lmdb_name, df_train, nmf_embedding)

    print('Predicting top categories for val.')
    tfidf = tfidf_vectorizer.transform([' '.join(item) for item in words_val])
    nmf_embedding = nmf_300.transform(tfidf)
    put_top_category_for_recipes(lmdb_name, df_val, nmf_embedding)

    print('Predicting top categories for test.')
    tfidf = tfidf_vectorizer.transform([' '.join(item) for item in words_test])
    nmf_embedding = nmf_300.transform(tfidf)
    put_top_category_for_recipes(lmdb_name, df_test, nmf_embedding)

###############below are the functions for w2v############################
# This function is modified based on https://www.kaggle.com/vukglisovic/classification-combining-lda-and-word2vec
""" Transform each recipe title into a feature vector. It averages out all the
    word vectors of recipe title.
    """
def get_w2v_features(w2v_model, words):
    words = words  # words in text
    index2word_set = set(w2v_model.wv.vocab.keys())  # words known to model

    featureVec = np.zeros(w2v_model.vector_size, dtype="float32")

    # Initialize a counter for number of words in a review
    nwords = 0
    # Loop over each word in the comment and, if it is in the model's vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            featureVec = np.add(featureVec, w2v_model[word])
            nwords += 1.

    # Divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

## layer_2 dict
def create_layer2_dict():
    print("create layer2 dict")
    layer2 = None
    with open('data/recipe1M/layer2.json') as f:
        layer2 = json.load(f)
    layer2_dict = {}
    for food in layer2:
        recipe_id = food.get("id")
        image_list = food.get("images")
        image_names = []
        for image in image_list:
            image_names.append(image["id"])
        layer2_dict[recipe_id] = image_names
    return layer2_dict

def create_image_recipe_dict_train(this_df, layer2_dict):
    img_recipe_dict = {}
    for i in range(len(this_df)):
        # print(i)
        recipe_id = this_df.iloc[i]["id"]
        recipe_title = this_df.iloc[i]["title_clean"]
        recipe_title_2 = this_df.iloc[i]["title_2"]
        recipe_topic_cluster = this_df.iloc[i]["topic_cluster"]
        # print(recipe_id)
        image_list = layer2_dict.get(recipe_id)
        # print(image_list)
        if image_list is None:
          continue
        for image in image_list:
            img_recipe_dict[image] = (recipe_id,recipe_title,recipe_title_2,recipe_topic_cluster)
        ## =========== remove this line or modify the number of recipes ==================
    return img_recipe_dict

def get_dataframe_all_info(df_all_info, dict_all):
    index = 0
    for key in dict_all:
        df_all_info.iloc[index]["Image name"] = key
        value = dict_all.get(key)
        df_all_info.iloc[index]["Recipe ID"] = value[0]
        df_all_info.iloc[index]["Title"] = value[1]
        df_all_info.iloc[index]["Recipe title pred"] = value[2]
        df_all_info.iloc[index]["Topic cluster"] = value[3]
        index += 1
    return df_all_info

def put_top_category_for_recipes(lmdb_name, df_title, w_vector):
    db = lmdb.open(lmdb_name, map_size=int(1e11))
    with db.begin(write=True) as txn:
        for i in tqdm(range(len(df_title))):
            one_hot_matrix = w_vector[i,:]
            if one_hot_matrix.any():
                category = np.argsort(one_hot_matrix)[::-1][0] # index of topics_category_300
            else: # all zeros
                category = -1 # no category
            recipe_id = df_title['id'][i]
            txn.put(recipe_id.encode('latin1'), str(category).encode('latin1'))

def build_w2v(n_topics):
    print("w2v starting")
    df_train, df_val, df_test = create_clean_titled_dfs()
    words_train = remove_stopwords(list(sentences_to_words(df_train['title_clean'].values.tolist())))
    words_val   = remove_stopwords(list(sentences_to_words(df_val['title_clean'].values.tolist())))
    words_test  = remove_stopwords(list(sentences_to_words(df_test['title_clean'].values.tolist())))

    num_features = n_topics    # Word vector dimensionality
    min_word_count = 3    # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 6           # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model
    W2Vmodel = Word2Vec(sentences=words_train,
                    sg=1,
                    hs=0,
                    workers=num_workers,
                    size=num_features,
                    min_count=min_word_count,
                    window=context,
                    sample=downsampling,
                    negative=5,
                    iter=6)

    print("w2v mapping")
    w2v_features = list(map(lambda sen_group: get_w2v_features(W2Vmodel, sen_group), words_train)) # train data
    w2v_trial = list(map(lambda sen_group: get_w2v_features(W2Vmodel, sen_group), words_val))  # val data
    w2v_test = list(map(lambda sen_group: get_w2v_features(W2Vmodel, sen_group), words_test))  # test data

    #Cluster by k-means https://medium.com/analytics-vidhya/topic-modelling-using-word-embeddings-and-latent-dirichlet-allocation-3494778307bc
    #1 hour to run
    print('K Means Clustering for train/val/test')
    km = KMeans(
        n_clusters=n_topics, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )

    y_km = km.fit_predict(w2v_features)
    df_w2v = pd.DataFrame({'title' :data_words, 'topic_cluster' :y_km })
    #df_w2v.to_csv("w2v_knn_cluster.csv")

    df_train_copy = copy.deepcopy(df_train)
    df_train_copy["title_2"] = df_w2v["title"]
    df_train_copy["topic_cluster"] = df_w2v["topic_cluster"]
    df_train_copy["w2v_features"] = w2v_features

    #predict cluster of test data
    cls_test2 = km.predict(w2v_trial)
    df_w2v_test = pd.DataFrame({'title' :data_words_test, 'topic_cluster' :cls_test2 })

    df_val_copy = copy.deepcopy(df_val)
    df_val_copy["title_2"] = df_w2v_test["title"]
    df_val_copy["topic_cluster"] = df_w2v_test["topic_cluster"]
    df_val_copy["w2v_features"] = w2v_trial

    #predict cluster of test data
    cls_test3 = km.predict(w2v_test)
    df_w2v_test2 = pd.DataFrame({'title' :data_words_test2, 'topic_cluster' :cls_test3 })

    df_test_copy = copy.deepcopy(df_test)
    df_test_copy["title_2"] = df_w2v_test2["title"]
    df_test_copy["topic_cluster"] = df_w2v_test2["topic_cluster"]
    df_test_copy["w2v_features"] = w2v_test

    layer2_dict = create_layer2_dict()

    print("create all info dataframe for train/val/test")
    dict_train = create_image_recipe_dict_train(df_train_copy, layer2_dict)
    df_all_info_train = pd.DataFrame(index = range(0, len(dict_train)), columns=['Image name','Recipe ID','Title','Recipe title pred','Topic cluster'])
    df_all_info_train = get_dataframe_all_info(df_all_info_train, dict_train)
    df_all_info_train = df_all_info_train.sort_values(by=["Topic cluster"])

    dict_val = create_image_recipe_dict_train(df_val_copy, layer2_dict)
    df_all_info_val = pd.DataFrame(index = range(0, len(dict_val)), columns=['Image name','Recipe ID','Title','Recipe title pred','Topic cluster'])
    df_all_info_val = get_dataframe_all_info(df_all_info_val, dict_val)
    df_all_info_val = df_all_info_val.sort_values(by=["Topic cluster"])

    dict_test = create_image_recipe_dict_train(df_test_copy, layer2_dict)
    df_all_info_test = pd.DataFrame(index = range(0, len(dict_test)), columns=['Image name','Recipe ID','Title','Recipe title pred','Topic cluster'])
    df_all_info_test = get_dataframe_all_info(df_all_info_test, dict_test)
    df_all_info_test = df_all_info_test.sort_values(by=["Topic cluster"])

    lmdb_name = f"data/w2v_{n_topics}_categories_lmdb"
    put_top_cluster_for_recipes(lmdb_name, df_all_info_train)
    put_top_cluster_for_recipes(lmdb_name, df_all_info_val)
    put_top_cluster_for_recipes(lmdb_name, df_all_info_test)

n_topics = int(sys.argv[2]) # number of topics
if sys.argv[1] == 'nmf':
    build_nmf(n_topics)
elif sys.argv[1] == 'w2v':
    build_w2v(n_topics)
else:
    print(f"unexpected topic modeling: {sys.argv[1]}")
