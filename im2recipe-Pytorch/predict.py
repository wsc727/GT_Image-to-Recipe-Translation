from PIL import Image
from args import get_parser
from trijoint import im2recipe
import numpy as np
import os
import pickle
import random
import sys
import torch
import torchfile
import torchvision.transforms as transforms
import subprocess
import simplejson as json

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

print('Loading recipe embeddings.')
with open(os.path.join(opts.path_results,'rec_embeds.pkl'),'rb') as f:
    instr_vecs = pickle.load(f)
with open(os.path.join(opts.path_results,'rec_ids.pkl'),'rb') as f:
    names = pickle.load(f)

if not(torch.cuda.device_count()):
    device = torch.device(*('cpu',0))
else:
    torch.cuda.manual_seed(opts.seed)
    device = torch.device(*('cuda',0))


# create model
model = im2recipe()
model.visionMLP = torch.nn.DataParallel(model.visionMLP)
model.to(device)

# load checkpoint
print("=> loading checkpoint '{}'".format(opts.model_path))
if device.type=='cpu':
    checkpoint = torch.load(opts.model_path, encoding='latin1', map_location='cpu')
else:
    checkpoint = torch.load(opts.model_path, encoding='latin1')
opts.start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})"
      .format(opts.model_path, checkpoint['epoch']))

def main(model, im_path):
    ext = os.path.basename(im_path).split('.')[-1]
    if ext not in ['jpeg','jpg','png']:
        raise Exception("Wrong image format.")

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256), # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224), # we get only the center of that rescaled
        transforms.ToTensor(),
        normalize,
    ])

    # load image
    im = Image.open(im_path).convert('RGB')
    im = transform(im)
    im = im.view((1,)+im.shape)
    # get model output
    output = model.visionMLP(im)
    output = output.view(output.size(0), -1)
    output = model.visual_embedding(output)
    output = norm(output)

    # calculate similarities
    sim = np.dot(output.cpu().data.numpy(), instr_vecs.T)[0]
    # sort indices in descending order
    sorting = np.argsort(sim)[::-1].tolist()
    # get recipe ids in descending order
    global names
    names = names[sorting]

    # just show only the most similar recipe for now
    recipe_id = names[0]

    # Find the recipe from layer1
    recipe = subprocess.check_output(['bash', '-c', f'cat data/recipe1M/layer1.json | grep "{recipe_id}"'])
    recipe = recipe.rstrip()[0:-1] # strip ",\n"
    recipe = json.loads(recipe)
    # from scripts.utils import Layer
    # layer1 = Layer.load('layer1', 'data/recipe1M') # slower than the above

    # Make it easier to read
    recipe['ingredients'] = [i['text'] for i in recipe['ingredients']]
    recipe['instructions'] = [i['text'] for i in recipe['instructions']]

    # Dump
    print(json.dumps(recipe, indent='  '))
    # print(f"{im_path}: [{recipe['title']}]({recipe['url']})")

main(model, opts.test_image_path)
# with open('../Food-Recipe-CNN/data/val100_images.pkl','rb') as f:
#     image_ids = pickle.load(f)
# for image_id in image_ids:
#     loader_path = os.path.join(*[image_id[i] for i in range(4)])
#     path = f"data/images/val/{loader_path}/{image_id}"
#     main(model, path)
