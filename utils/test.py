import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score
from utils.define_fct import build_fct

from utils.define_unet import build_unet
from utils.prepare_data import create_dir, seeding

def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)
    score_auc = roc_auc_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_auc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

def model_performance(modArch):
    """ Seeding """
    seeding(31)

    """ Folders """
    create_dir("results")

    """ Load dataset """
    test_x = sorted(glob("C:/Users/Derek/Documents/SLU_Capstone/new_data/test/image/*"))
    test_y = sorted(glob("C:/Users/Derek/Documents/SLU_Capstone/new_data/test/mask/*"))

    """ Hyperparameters """
    if modArch == 'unet':
        H = 512
        W = 512
        checkpoint_path = "files/checkpoint_unet.pth"
    if modArch == 'fct':
        H = 256
        W = 256
        checkpoint_path = "files/checkpoint.pth"
    size = (W, H)

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if modArch == 'unet':
        model = build_unet()
    else:
        model = build_fct()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = x.split('/')[-1].split('\\')[-1].split('.')[0]

        """ Reading image """
        if modArch == 'unet':
            image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
            x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
            x = x/255.0
            x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
            x = x.astype(np.float32)
            x = torch.from_numpy(x)
            x = x.to(device)
        if modArch == 'fct':
            image = cv2.imread(x, cv2.IMREAD_COLOR)
            # Shrink Image for VRAM Limitations
            scale_percent = 0.5
            width = int(image.shape[1]*scale_percent)
            height = int(image.shape[0]*scale_percent)
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            x = np.transpose(image, (2, 0, 1))      ## (3, 256, 256)
            x = x/255.0
            x = np.expand_dims(x, axis=0)           ## (1, 3, 256, 256)
            x = x.astype(np.float32)
            x = torch.from_numpy(x)
            x = x.to(device)

        """ Reading mask """
        if modArch == 'unet':
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
            y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
            y = y/255.0
            y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
            y = y.astype(np.float32)
            y = torch.from_numpy(y)
            y = y.to(device)
        if modArch == 'fct':
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
            # Shrink Image for VRAM Limitations
            scale_percent = 0.5
            width = int(mask.shape[1]*scale_percent)
            height = int(mask.shape[0]*scale_percent)
            dim = (width, height)
            mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
            y = np.expand_dims(mask, axis=0)            ## (1, 256, 256)
            y = y/255.0
            y = np.expand_dims(y, axis=0)               ## (1, 1, 256, 256)
            y = y.astype(np.float32)
            y = torch.from_numpy(y)
            y = y.to(device)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            if modArch == 'unet':
                pred_y = model(x)
                pred_y = torch.sigmoid(pred_y)
            else:
                pred_y = model(x)
                #pred_y = torch.sigmoid(pred_y[2])
            total_time = time.time() - start_time
            time_taken.append(total_time)

            if modArch == 'unet':
                score = calculate_metrics(y, pred_y)
                metrics_score = list(map(add, metrics_score, score))
                pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
                pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
                pred_y = pred_y > 0.5
                pred_y = np.array(pred_y, dtype=np.uint8)
            else:
                score = calculate_metrics(y, pred_y[2])
                metrics_score = list(map(add, metrics_score, score))
                pred_y = pred_y[2][0].cpu().numpy()        ## (1, 256, 256)
                pred_y = np.squeeze(pred_y, axis=0)     ## (256, 256)
                pred_y = pred_y > 0.5
                pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        if modArch == 'unet':
            cat_images = np.concatenate(
                [image, line, ori_mask, line, pred_y * 255], axis=1
            )
            cv2.imwrite(f"results/{name}_unet.png", cat_images)
        else:
            cat_images = np.concatenate(
                [image, line, ori_mask, line, pred_y * 255], axis=1
            )
            cv2.imwrite(f'results/{name}_fct.png', cat_images)

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    auc = metrics_score[5]/len(test_x)
    print(f"Jaccard: {jaccard:1.4f}")
    print('')
    print(f'F1: {f1:1.4f}')
    print('')
    print(f'Precision: {precision:1.4f}')
    print('')
    print(f'Recall: {recall:1.4f}')
    print('')
    print(f'Accuracy: {acc:1.4f}')
    print('')
    print(f'AUC ROC: {auc:1.4f}')


    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)