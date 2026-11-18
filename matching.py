# imports
import glob
import os
from tqdm import tqdm
import torch

import pickle
from triplet.triplet_main import test_LG
from loma import LoMa, LoMaR

def save_features(features, path = "noAnnotResorig"):
    for f in tqdm(features):
        serialized = pickle.dumps(features[f])
        fil = open("feats/"+f.split("/")[1]+".val", "wb")
        fil.write(serialized)
        fil.close()
    
    load_features(features)

def save_feature(f, name, path = "noAnnotResorig"):
    serialized = pickle.dumps(f)
    fil = open("feats/"+name.split("/")[1]+".val", "wb")
    fil.write(serialized)
    fil.close()


def load_features(name, path = "feats"):
    fil = open(path+"/"+name.split("/")[len(name.split("/"))-1]+".val", "rb")
    data = fil.read()
    feature = pickle.loads(data)
    return feature

# loads database
def load_database(path = "noAnnotResorig"):

    # feature extractor
    #extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  
    #extractor = SIFT(max_num_keypoints=2048).eval().cuda()  
    matcher = LoMa(LoMaR())

    # all images .jpg in repo
    img_files2 = glob.glob(os.path.join(path,'*.jpg'))
    features = {}

    os.makedirs("feats", exist_ok=True)

    print("extracting features from database")
    for i in tqdm(sorted(img_files2)):
        # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]

        if not os.path.exists("feats/"+i.split("/")[len(i.split("/"))-1]+".val"):

            print(i)
            # extract local features
            feats1 =  matcher.extract(i)

            feats2 = {}
            # convert to np to keep on cpu, preventing multiple computations of features for one image
            feats2 = [k.cpu().data.numpy() for k in feats1]
            features[i] = feats2

            save_feature(features[i], i)
        else:
            features[i] = load_features(i)

    return features          

def match_repository(features, im, path):
    
    matcher = LoMa(LoMaR())
    
    # all .jpg
    print("matching")

    img_file = im

    thr = 300

    maxP = 0
    nameP = ""

    args = ["RegNet", "logs", "op1p2p3", 224, 64, "noAnnotResorig"]

    dists, name_dists = test_LG(args, img_file)

    k_pred = 0
    k_first = len(features)

    #print(len(imgs))
    A = matcher.extract(path)

    #for img in imgs:
    for index in tqdm(range(k_pred, k_first, 1)):
        i = name_dists[int(dists[index][1])]
        if(maxP<thr):#max_score<thr):
            # get database features
            B = [torch.tensor(k).cuda() for k in features[i]]
            #print(i)

            kptsA = matcher.match_descr(A, B)

            if(kptsA>maxP):
                maxP=kptsA
                nameP = i
        if(maxP>thr):
            break

    return (nameP, maxP)

