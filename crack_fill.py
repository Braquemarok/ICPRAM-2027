#POUR LES CAROTTES DE L'INRAE 

import logging
import os
import sys

import numpy as np
import glob
import math
from tqdm import tqdm
import numpy as np
import torch
from ultralytics import YOLO

import cv2

detect = "detection"
bilat = "bilat"
thresh = "thr"
f_res_dir = "res"

def yolo_sol(patches_o, img, mask, psize, overlap, path):

    os.makedirs("results_after_yolo", exist_ok=True)

    #print("##############\n##############\n############")
    x1, x2, y1, y2 = getInds(mask)
    lenX = x2-x1
    lenY = y2-y1
    index = 0
    mask2 = np.zeros(mask.shape).astype(np.uint8)
    mask3 = np.zeros(mask.shape).astype(np.uint8)
    mask4 = np.zeros(mask.shape).astype(np.uint8)

    # Use the model
    model = YOLO("best.pt")
    #model.conf = 0.00005
    patches = []

    ind = 0
    for imgp in tqdm(patches_o):
        #if not os.path.isfile(imgp.replace(".jpg", ".png").replace("org", "results"+i)):
            #img= cv2.imread(imgp, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            results = model.predict(source=imgp.copy(), save=False, save_txt=False, stream=True, verbose=False)
            for result in results:
                if result.masks is not None:
                    # get array results
                    #print(imgp)
                    #print(imgp.replace(".jpg", ".png").replace("org", "results"+i))
                    masks = result.masks.data
                    boxes = result.boxes.data
                    # extract classes
                    clss = boxes[:, 5]
                    # get indices of results where class is 0 (people in COCO)
                    people_indices = torch.where(clss == 0)
                    # use these indices to extract the relevant masks
                    people_masks = masks[people_indices]
                    # scale for visualizing results
                    people_mask = torch.any(people_masks, dim=0).int()
                    # save to file
                    res = cv2.resize(people_mask.cpu().numpy().astype(np.uint8), dsize=(imgp.shape[1], imgp.shape[0]), interpolation=cv2.INTER_CUBIC)
                    patches.append(res)
                else:
                    patches.append(np.zeros((imgp.shape[0], imgp.shape[1])))
            #cv2.imwrite("patches_MMR/"+str(ind)+".jpg", p)
            ind+=1

    for x in range(x1, x1+lenX, psize-overlap):
        for y in range(y1, y1+lenY, psize-overlap):

            p = mask[x:x+psize, y:y+psize]

            cols = p.sum(axis=1)
            rows = p.sum(axis=0)

            verif = cols[0]+cols[len(cols)-1]+rows[0]+rows[len(rows)-1]
            
            if(verif>=255*psize*4):
                mt = mask2[x:x+psize, y:y+psize]

                #print("####################")
                #print(pt.shape)
                #print(patches[index].shape)
                mt[patches[index]>0] = (1)
                mt2 = mask3[x:x+psize, y:y+psize]
                mt2[patches[index]>0] = (255)
                index+=1


    cv2.imwrite("results_after_yolo/"+path.split(".")[0]+".png", mask3)

    return mask3

# return starting and ending indexes of masked object
def getInds(mask):
    cols = mask.sum(axis=1)
    rows = mask.sum(axis=0)

    firstx = -1
    lastx = -1
    firsty = -1
    lasty = -1

    for x in range(len(cols)):
        if(cols[x]>0):
            if firstx<0:
                firstx = x
            else:
                lastx = x

    for y in range(len(rows)):
        if(rows[y]>0):
            if firsty<0:
                firsty = y
            else:
                lasty = y  

    return (firstx, lastx, firsty, lasty)

# return patches from the masked object
def get_patches(img, mask, psize, overlap):

    os.makedirs("patches", exist_ok=True)
    os.makedirs("patches_MASK", exist_ok=True)

    x1, x2, y1, y2 = getInds(mask)
    lenX = x2-x1
    lenY = y2-y1
    patches = []
    patches_M = []

    for x in range(x1, x1+lenX, psize-overlap):
        for y in range(y1, y1+lenY, psize-overlap):

            p = mask[x:x+psize, y:y+psize]

            cols = p.sum(axis=1)
            rows = p.sum(axis=0)

            verif = cols[0]+cols[len(cols)-1]+rows[0]+rows[len(rows)-1]
            
            if(verif>=255*psize*4):
                #print(verif)
                patches.append(img[x:x+psize, y:y+psize].copy())
                patches_M.append(p)
    
    for i in range(len(patches)):
        cv2.imwrite("patches/"+str(i)+".jpg", patches[i])
        cv2.imwrite("patches_MASK/"+str(i)+".jpg", patches_M[i])
    
    return (patches, patches_M)

# takes the minimal components from the object (hopefully slits)
def get_anomalies_per_patch(patches, patches_M):
    assert(len(patches)==len(patches_M))
    os.makedirs("patches_M", exist_ok=True)
    os.makedirs("patches_PR", exist_ok=True)
    os.makedirs("patches_MR", exist_ok=True)
    os.makedirs("patches_MMR", exist_ok=True)
    os.makedirs("patches_Hard_R", exist_ok=True)
    for i in range(len(patches)):

        # recup patches with slits and shit
        vec = patches[i].ravel()
        vecG = patches[i].copy()

        vecG = cv2.cvtColor(vecG, cv2.COLOR_BGR2GRAY)
        vecG = vecG.ravel()

        std_vec = np.std(vec)
        min_vec = np.min(vec)

        std_vecG = np.std(vecG)
        min_vecG = np.min(vecG)

        p = patches[i]
        #p = cv2.bilateralFilter(p, 9, 75, 75)

        min_shiiih = 20
        std_vecG = min(std_vecG, min_shiiih)

        p[p>int(min_vec+std_vec)]=0
        p[p>int(min_vecG+std_vecG)]=0
        p[p>0]=255

        #print(np.unique(p))

        cv2.imwrite("patches_M/"+str(i)+".jpg", p) # super beau résultats, différencier cas général clair/foncé

        p = cv2.imread("patches_M/"+str(i)+".jpg")

        pR = p[:,:,2]
        pB = p[:,:,0]
        pG = p[:,:,1]

        pR[(pR<=pB) | (pR<=pG)]=0
        pB[(pB<=pR)]=0
        pG[pG>0]=0

        vec_R = pR.copy().ravel()
        vec_R = vec_R[vec_R!=0]
        if(vec_R.shape[0]>0):
            std_vec_R = np.std(vec_R)
            max_vec_R = np.max(vec_R)

            pR[pR<max_vec_R-std_vec_R*2]=0
            pR[pR>0]=255

            if(np.sum(pR)//255<40):
                pR[pR>0]=0
                vec_B = pB.copy().ravel()
                vec_B = vec_B[vec_B!=0]

                if(vec_B.shape[0]>0):
                    std_vec_B = np.std(vec_B)
                    max_vec_B = np.max(vec_B)

                    pB[pB<max_vec_B-std_vec_B*2]=0

                    pG[pB>0]=255
            pB[pB>0]=0
        
        cv2.imwrite("patches_PR/"+str(i)+".jpg", p)

        p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)

        p[p>0]=255
        k = np.ones((7, 7), np.uint8)
        p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, k)

        unique, counts = np.unique(p, return_counts=True)
        t_d = dict(zip(unique, counts))

        #print("#############")
        #print(i)
        #print(str(t_d[255]/(t_d[0]+t_d[255])))

        # about 20% for big poopoo
        if(255 in t_d and 0 in t_d):
            if(t_d[255]/(t_d[0]+t_d[255])>0.2):
                p[p>0]=0
            else:
                """output = cv2.connectedComponentsWithStats(p, 8, cv2.CV_32S)
                
                labels = output[1]
                stats = output[2]

                mini = 0
                minL = 0
                pi = {}
                pe = []

                for nis in range(1, len(stats), 1):
                    pi[nis] = stats[nis][4]
                    pe.append(stats[nis][4])
                    if(stats[nis][4]<mini):
                        mini = stats[nis][4]
                        minL = nis
                
                pi = {k: v for k, v in sorted(pi.items(), key=lambda item: item[1])}
                #print(pi)
                pe = np.array(pe)
                #print(pe)
                #print(pi)
                pe_std = np.std(pe)

                piname = None
                pival = 0
                #print(pe_std)
                pidel = []

                for nis in pi:
                    if piname is None:
                        piname = nis
                        pival = pi[piname]

                    if pi[nis]>max(pival+pe_std, 60):
                        pidel.append(nis)

                for nis in pidel:
                    del pi[nis]
                #print(pi)
                for nis in pi:
                    p[labels==nis]=0"""
                pass

            cv2.imwrite("patches_MR/"+str(i)+".jpg", p)
        elif(0 not in t_d):
            p[p>0] = 0
        cv2.imwrite("patches_MMR/"+str(i)+".jpg", p)
        patches[i] = p

    #exit()
    return patches

# return patches from the masked object
def set_patches(img, mask, psize, overlap, patches, path):

    os.makedirs("results_after", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results_before", exist_ok=True)

    #print("##############\n##############\n############")
    x1, x2, y1, y2 = getInds(mask)
    lenX = x2-x1
    lenY = y2-y1
    index = 0
    mask2 = np.zeros(mask.shape).astype(np.uint8)
    mask3 = np.zeros(mask.shape).astype(np.uint8)
    mask4 = np.zeros(mask.shape).astype(np.uint8)

    for x in range(x1, x1+lenX, psize-overlap):
        for y in range(y1, y1+lenY, psize-overlap):

            p = mask[x:x+psize, y:y+psize]

            cols = p.sum(axis=1)
            rows = p.sum(axis=0)

            verif = cols[0]+cols[len(cols)-1]+rows[0]+rows[len(rows)-1]
            
            if(verif>=255*psize*4):
                pt = img[x:x+psize, y:y+psize]
                mt = mask2[x:x+psize, y:y+psize]

                #print("####################")
                #print(pt.shape)
                #print(patches[index].shape)
                mt[patches[index]>0] = (1)
                mt2 = mask3[x:x+psize, y:y+psize]
                mt2[patches[index]>0] = (255)

                pt2 = cv2.cvtColor(pt, cv2.COLOR_BGR2GRAY)
                m_pt = np.ma.masked_array(pt2, mask=~mt.astype(bool))
                std_vec = np.std(m_pt)
                average_vec = np.average(m_pt)
                min_vec = np.min(m_pt)

                # that shit mt[(pt2<average_vec+std_vec/2) & (pt2>average_vec-std_vec/2)] = 1
                # that sucksmt[pt2<average_vec-std_vec/2] = 0
                mt[(pt2<=min_vec+std_vec/2) & (pt2>0)] = 1
                index+=1

                mt[mt>0]=255

    mask4 = mask2.copy()

    output = cv2.connectedComponentsWithStats(mask4, 8, cv2.CV_32S)
            
    labels = output[1]
    stats = output[2]

    pi = {}
    pe = []

    for nis in range(1, len(stats), 1):
        pi[nis] = stats[nis][4]
        pe.append(stats[nis][4])
                
    pi = {k: v for k, v in sorted(pi.items(), key=lambda item: item[1])}
    #print(pi)
    pe = np.array(pe)
    pe_std = np.std(pe)
    piname = None
    pival = 0
    #print(pe_std)
    pidel = []

    for nis in pi:
        if piname is None:
            piname = nis
            pival = pi[piname]

        if pi[nis]>pival+pe_std/2:
            pidel.append(nis)

    for nis in pidel:
        del pi[nis]

    #print(pi)

    # needs GPU 
    mask4 = torch.from_numpy(mask4).cuda()
    labels = torch.from_numpy(labels).cuda()

    for nis in pi:
        mask4[labels==nis]=0
        #labels[labels==nis]=0

    """labels_np = labels.cpu().numpy()

    print(np.unique(labels_np))

    masks=[]

    for m in np.unique(labels_np):
        if m!=0:
            yikes = np.zeros(labels.shape)
            yikes[labels==m]=255
            masks.append()"""

    mask4 = mask4.cpu().numpy()

    contours, _ = cv2.findContours(mask4, 1, 2)

    mask4 = cv2.cvtColor(mask4, cv2.COLOR_GRAY2BGR)
    
    #print(len(contours))
    for cnt in contours:
        if(len(cnt)>=5):
            rect = cv2.minAreaRect(cnt)
            #print(rect[1][0]/rect[1][1])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if rect[1][0]/rect[1][1] < 1.7 and rect[1][0]/rect[1][1] > .3 and (rect[1][0]<100) or (min(rect[1][0], rect[1][1])>20 and max(rect[1][0], rect[1][1])<100):  
                cv2.drawContours(mask4,[box],0,(0,0,0),-1)
                #cv2.drawContours(mask4,[box],0,(0,0,255),2)
            #else:
                #cv2.drawContours(mask4,[box],0,(0,255,0),2)

    #k = np.ones((10, 10), np.uint8)
    #mask4 = cv2.morphologyEx(mask4, cv2.MORPH_CLOSE, k)
    
    cv2.imwrite("results/"+path.split(".")[0]+".png", mask2)
    cv2.imwrite("results_before/"+path.split(".")[0]+".png", mask3)
    cv2.imwrite("results_after/"+path.split(".")[0]+".png", mask4)

    return mask4


def inference(img_path, resMask = "testMask/", notMask = "testRes/", vr = "visualerror/"):
    ''' 
    effectue les inférences sur l'ensemble d'une image
    '''
    logging.info("Processing a single frame")

    p_size = 128
    overlap = int(p_size//4)#p_size//2

    # chargement de l'image
    data = cv2.imread(img_path)
    #mask = cv2.imread("bw/"+img_path.split("/")[len(img_path.split("/"))-1].split(".")[0]+".png")
    mask = np.ones(data.shape).astype(np.uint8)*255
    #data = cv2.bilateralFilter(data, 25, 125, 125)
    data_G = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    if(len(mask.shape)>2):
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    patches, patches_M = get_patches(data, mask, p_size, overlap)

    deep = 1
    if(deep == 0):
        patches = get_anomalies_per_patch(patches, patches_M)
        mask_K = set_patches(data, mask, p_size, overlap, patches, img_path.split("/")[len(img_path.split("/"))-1])
    elif(deep==1):
        mask_K = yolo_sol(patches, data, mask, p_size, overlap, img_path.split("/")[len(img_path.split("/"))-1])
    else:
        if os.path.exists("refs/"+img_path.split("/")[len(img_path.split("/"))-1].split(".")[0]+".png"):
            mask_K = cv2.imread("refs/"+img_path.split("/")[len(img_path.split("/"))-1].split(".")[0]+".png")
        else:
            return None

    #exit()
    k = np.ones((4, 4), np.uint8)
    mask_K = cv2.morphologyEx(mask_K, cv2.MORPH_DILATE, k)

    k = np.ones((5, 5), np.uint8)
    dataF = cv2.morphologyEx(data, cv2.MORPH_OPEN, k)

    k = np.ones((20, 20), np.uint8)
    dataF = cv2.morphologyEx(dataF, cv2.MORPH_DILATE, k)

    data[mask_K>0] = 0
    dataF[mask_K==0] = 0
    data+=dataF
    
    cv2.imwrite(thresh+"/"+img_path.split("/")[len(img_path.split("/"))-1], data)

    #exit()
    return data

def corpsDoss(path):
    '''
    programme principal pour un dossier
    '''
    logging.info("Inference")
    folder_path = path+"/"
    img_files = sorted(glob.glob(os.path.join(folder_path,'','*.jpg')))

    i = 0
    for img_path in tqdm(img_files):
        #print("inférence sur "+img_path)
        if i>800:
            inf = inference(img_path).astype(np.uint8)
        i+=1
        #print("post_traitement")

def corpsFile(img_path):
    '''
    programme principal pour un fichier
    '''
    print("inférence sur "+img_path)
    inf = inference(img_path).astype(np.uint8)
    print("post_traitement")

if __name__ == "__main__":
    #torch.cuda.init()
    if not torch.cuda.is_available():
        print("NO CUDA DEVICE AVAILABLE")
        print("BYE")
        exit()

    os.makedirs(detect, exist_ok=True)
    os.makedirs(bilat, exist_ok=True)
    os.makedirs(f_res_dir, exist_ok=True)
    os.makedirs(thresh, exist_ok=True)

    # test2()
    if len(sys.argv)<2:
        print("problème dans les paramètres, pour utiliser ce programme, \n inference6.py <dossier/fichier à traiter>")
        exit(0)
    
    if(".jpg" in sys.argv[1]):
        corpsFile(sys.argv[1])
    else:
        if(".png" in sys.argv[1]):
            print("problème dans les paramètres, l'image doit être au format .tif")
        corpsDoss(sys.argv[1])
