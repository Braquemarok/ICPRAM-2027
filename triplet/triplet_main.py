import os
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import glob
import pickle

def model_builder(model_name, out_dim):
    if(model_name=="RegNet"):
        from torchvision.models import regnet_y_3_2gf, RegNet_Y_3_2GF_Weights
        model = regnet_y_3_2gf(RegNet_Y_3_2GF_Weights.IMAGENET1K_V2)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_dim)
    else:
        exit()
    return model
    
class RegNet(nn.Module):
    def __init__(self, args, pretrained=True):
        super(RegNet, self).__init__()
        self.reg= model_builder(args[0], args[4])

    def forward(self, x):
        x = self.reg(x)
        return x


AVAILABLE_MODELS = {"RegNet" : RegNet}

def save_feature(f, name):
    serialized = pickle.dumps(f)
    fil = open("triplet_feats/"+name.split("/")[1]+".val", "wb")
    fil.write(serialized)
    fil.close()


def load_features(name, path = "triplet_feats"):
    fil = open(path+"/"+name.split("/")[len(name.split("/"))-1]+".val", "rb")
    data = fil.read()
    feature = pickle.loads(data)
    return feature  

def test_LG(args, test_img):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = []
    embeddings = []
    session2_path = []
    distances = {}
    names_distances = {}
    ids_distances = {}
    kluster_size = 500
    nb_k = 1
    ids = []
    
    # loads regnet
    os.makedirs("triplet_feats", exist_ok=True)

    model = AVAILABLE_MODELS[args[0]](args).to(device)
    model.load_state_dict(torch.load(os.path.join(args[1], args[2], "best_model.pth")))
    
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((args[3], args[3])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        for img in sorted(glob.glob(os.path.join(args[5], "*.jpg"))):
            if("R" not in img.split("_")[len(img.split("_"))-1]):
                if not os.path.exists("triplet_feats/"+img.split("/")[len(img.split("/"))-1]+".val"):
                    im = Image.open(img)
                    im = data_transforms(im)
                    tim = torch.FloatTensor(im).cuda()
                    tim = tim.unsqueeze(0)
                    feature = model(tim).cpu().numpy()
                    embeddings.append(feature)
                    save_feature(feature, img)
                else:
                    feature = load_features(img)
                    embeddings.append(feature)
                names.append(img)
                ids.append(img.split("/")[len(img.split("/"))-1].replace(img.split("_")[len(img.split("_"))-1], ""))
            else:
                session2_path.append(img)

        name = ""
        for img in sorted([test_img]):
            im = Image.fromarray(img)
            im = data_transforms(im)
            tim = torch.FloatTensor(im).cuda()
            tim = tim.unsqueeze(0)
            #print(tim.shape)
            emb = model(tim)
            distances[name] = []
            names_distances[name] = []
            ids_distances[name] = []
            for i in range(len(embeddings)):
                e2 = torch.Tensor(embeddings[i]).cuda()
                dist = F.pairwise_distance(emb, e2, 2)
                distances[name].append([float(dist.cpu().numpy()[0]), i])
                names_distances[name].append(names[i])
                ids_distances[name].append(ids[i])
            distances[name] = np.asarray(distances[name], dtype=np.float32)


        #kluster_size = min(kluster_size, len(distances[name]))
        for key_img in distances:
            #print(distances[key_img])
            distances[key_img] = distances[key_img][distances[key_img][:, 0].argsort()]
            #print(distances[key_img])
 
        return distances[key_img], names_distances[key_img]
