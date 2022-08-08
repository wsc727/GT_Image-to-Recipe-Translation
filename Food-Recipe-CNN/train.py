# Fine-tune a pre-trained InceptionV3. Other models like Topic Modeling and IPCA/ANN aren't trained here.
from PIL import Image
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm
import copy
import lmdb
import numpy as np
import os
import pickle
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

class LabelLoader(Dataset):
    def __init__(self, transform, partition, topic_modeling, n_topics):
        self.partition = partition
        self.transform = transform
        self.recipe_db = lmdb.open(f"data/{partition}_lmdb", max_readers=1,
                                   readonly=True, lock=False, readahead=False, meminit=False)
        self.label_db  = lmdb.open(f"data/{topic_modeling}_{n_topics}_categories_lmdb", max_readers=1,
                                   readonly=True, lock=False, readahead=False, meminit=False)
        with open(f"data/{partition}_keys.pkl", 'rb') as f:
            self.ids = pickle.load(f)

    def __getitem__(self, index):
        with self.recipe_db.begin(write=False) as txn:
            recipe = txn.get(self.ids[index].encode('latin1'))

        # Select an image
        imgs = pickle.loads(recipe, encoding='latin1')['imgs']
        if self.partition == 'train':
            # As per im2recipe, we do only use the first five images per recipe during training
            img_idx = np.random.choice(range(min(5, len(imgs))))
        else:
            img_idx = 0

        # Load the image
        loader_path = os.path.join(*[imgs[img_idx]['id'][i] for i in range(4)])
        path = f"data/images/{self.partition}/{loader_path}/{imgs[img_idx]['id']}"
        try:
            img = self.transform(Image.open(path).convert('RGB'))
        except FileNotFoundError as e:
            img = torch.zeros([3, 299, 299]) # filter_inputs filters this out

        # Load the category
        with self.label_db.begin(write=False) as txn:
            label = txn.get(self.ids[index].encode('latin1'))
            label = -1 if label is None else int(label) # w2v's LMDB didn't write recipe ids missing images

        return img, label

    def __len__(self):
        return len(self.ids)

def initialize_incv3(num_classes):
    model_inc = models.inception_v3(pretrained=True)
    # We only update the reshaped layer params
    # ref: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    for param in model_inc.parameters():
        param.requires_grad = False

    # Handle the auxilary net
    model_inc.AuxLogits.fc = nn.Linear(model_inc.AuxLogits.fc.in_features, num_classes)
    # Handle the primary net
    model_inc.fc = nn.Linear(model_inc.fc.in_features, num_classes)
    return model_inc

# Filter out label=-1 (no_cat) and inexistent images
def filter_inputs(inputs, labels):
    idxs = (labels >= 0).nonzero(as_tuple=True)
    inputs = inputs[idxs]
    labels = labels[idxs]

    idxs = inputs.view([len(inputs), -1]).sum(1).nonzero(as_tuple=True)
    inputs = inputs[idxs]
    labels = labels[idxs]
    return inputs, labels

def train_model(model, data_loaders, criterion, optimizer, topic_modeling, n_topics, num_epochs=30):
    since = time.time()
    best_acc = 0.0

    model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(data_loaders[phase]):
                inputs = inputs.to(device).type(torch.cuda.FloatTensor)
                labels = torch.squeeze(labels).to(device)

                inputs, labels = filter_inputs(inputs, labels)
                if len(inputs) <= 1: # model(inputs) expects a batch of inputs (len > 1)
                    print('skipping an invalid batch.')
                    continue

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                state = { 'state_dict': model.state_dict() }
                torch.save(state, 'snapshots/%s_%d_model_e%03d_v-%.3f.pth.tar' % (topic_modeling, n_topics, epoch, best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

def main():
    topic_modeling = sys.argv[1]
    n_topics = int(sys.argv[2])

    # Initialize the model for this run
    InceptionV3 = initialize_incv3(n_topics)

    # Parameter for model hyperparameter tuning
    learning_rate = 0.001
    momentum_value = 0.9
    criterion = nn.CrossEntropyLoss()
    num_epochs = 30

    # Prepare an optimizer
    params_to_update = [param for _, param in InceptionV3.named_parameters()
                        if param.requires_grad == True]
    optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum_value)

    # Prepare data loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_loaders = {
        'train': DataLoader(
            dataset=LabelLoader(
                transform=transforms.Compose([
                    transforms.Scale(320), # rescale the image keeping the original aspect ratio
                    transforms.CenterCrop(320), # we get only the center of that rescaled
                    transforms.RandomCrop(299), # random crop within the center crop 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                partition='train', topic_modeling=topic_modeling, n_topics=n_topics,
            ),
            batch_size=25, num_workers=8, shuffle=True, pin_memory=True,
        ),
        'val': DataLoader(
            dataset=LabelLoader(
                transform=transforms.Compose([
                    transforms.Scale(320), # rescale the image keeping the original aspect ratio
                    transforms.CenterCrop(299), # we get only the center of that rescaled
                    transforms.ToTensor(),
                    normalize,
                ]),
                partition='val', topic_modeling=topic_modeling, n_topics=n_topics,
            ),
            batch_size=25, num_workers=8, shuffle=False, pin_memory=True,
        ),
    }

    # Train and evaluate
    train_model(InceptionV3, data_loaders, criterion, optimizer, topic_modeling, n_topics, num_epochs=num_epochs)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()
