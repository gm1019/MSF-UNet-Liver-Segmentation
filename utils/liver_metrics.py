# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:35:56 2024

@author: Ming Gong
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:22:10 2019

@author: xgb15139
"""
import numpy as np
import os
import cv2
import glob
from data import compute_iou
from scipy import ndimage
import GeodisTK

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score


def compute_dice(label_img, pred_img, p_threshold=0.5):
    p = pred_img.astype(np.float64)
    l = label_img.astype(np.float64)
    if p.max() > 127:
        p /= 255.
    if l.max() > 127:
        l /= 255.

    p = np.clip(p, 0, 1.0)
    l = np.clip(l, 0, 1.0)
    p[p > 0.5] = 1.0
    p[p < 0.5] = 0.0
    l[l > 0.5] = 1.0
    l[l < 0.5] = 0.0
    product = np.dot(l.flatten(), p.flatten())
    dice_num = 2 * product + 1
    pred_sum = p.sum()
    label_sum = l.sum()
    dice_den = pred_sum + label_sum + 1
    dice_val = dice_num / dice_den
    return dice_val 


def get_jaccard_index(label_img,pred_img):
    p = pred_img.astype(np.float64)
    l = label_img.astype(np.float64)
    if p.max() > 127:
        p /= 255.
    if l.max() > 127:
        l /= 255.

    p = np.clip(p, 0, 1.0)
    l = np.clip(l, 0, 1.0)
    p[p > 0.5] = 1.0
    p[p < 0.5] = 0.0
    l[l > 0.5] = 1.0
    l[l < 0.5] = 0.0
    product = np.dot(l.flatten(), p.flatten())
    dice_num = product 
    y = np.bitwise_or(p.astype(int),l.astype(int))
    dice_den = np.sum(y)
    dice_val = dice_num / dice_den
    return 1-dice_val 


def compute_rvd(label_img,pred_img):
    p = pred_img.astype(np.float64)
    l = label_img.astype(np.float64)
    if p.max() > 127:
        p /= 255.
    if l.max() > 127:
        l /= 255.

    p = np.clip(p, 0, 1.0)
    l = np.clip(l, 0, 1.0)
    p[p > 0.5] = 1.0
    p[p < 0.5] = 0.0
    l[l > 0.5] = 1.0
    l[l < 0.5] = 0.0
    rvd = abs((p.sum()-l.sum()))/l.sum()
    
    return rvd


def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim == 2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge

def binary_assd(s, g, spacing=None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd

def binary_hausdorff95(s, g, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)





def Rmse(pred_img,label_img):
    p = pred_img.astype(np.float64)
    l = label_img.astype(np.float64)
    if p.max() > 127:
        p /= 255.
    if l.max() > 127:
        l /= 255.

    p = np.clip(p, 0, 1.0)
    l = np.clip(l, 0, 1.0)
    p[p > 0.5] = 1.0
    p[p < 0.5] = 0.0
    l[l > 0.5] = 1.0
    l[l < 0.5] = 0.0
    rmse = np.sqrt(np.sum(np.power(p-l,2)/pow(len(p),2)))
    
    return rmse
    

def recall(pred_img,label_img):
    p = pred_img.astype(np.float64)
    l = label_img.astype(np.float64)
    if p.max() > 127:
        p /= 255.
    if l.max() > 127:
        l /= 255.

    p = np.clip(p, 0, 1.0)
    l = np.clip(l, 0, 1.0)
    p[p > 0.5] = 1.0
    p[p < 0.5] = 0.0
    l[l > 0.5] = 1.0
    l[l < 0.5] = 0.0
    GT_pos_sum = np.sum(l == 1)
#统计预测的mask中正样本的个数
    Mask_pos_sum = np.sum(p == 1)
#统计在groundtruth和mask相同位置都是正样本的个数，即实际为正样本，预测也是正样本的个数
    True_pos_sum = np.sum((l == 1) * (p == 1))
#那么实际为正样本，预测也为正样本占预测的mask中正样本的比例就是Precision
    Precision = float(True_pos_sum) / (Mask_pos_sum + 1e-6)
#实际为正样本，预测也为正样本占groundtruth中正样本的比例就是Recall
    Recall = float(True_pos_sum) / (GT_pos_sum + 1e-6)
    
    return Recall


def percision(pred_img,label_img):
    p = pred_img.astype(np.float64)
    l = label_img.astype(np.float64)
    if p.max() > 127:
        p /= 255.
    if l.max() > 127:
        l /= 255.

    p = np.clip(p, 0, 1.0)
    l = np.clip(l, 0, 1.0)
    p[p > 0.5] = 1.0
    p[p < 0.5] = 0.0
    l[l > 0.5] = 1.0
    l[l < 0.5] = 0.0
    
#统计ground truth中正样本的个数
    GT_pos_sum = np.sum(l == 1)
#统计预测的mask中正样本的个数
    Mask_pos_sum = np.sum(p == 1)
#统计在groundtruth和mask相同位置都是正样本的个数，即实际为正样本，预测也是正样本的个数
    True_pos_sum = np.sum((l == 1) * (p == 1))
#那么实际为正样本，预测也为正样本占预测的mask中正样本的比例就是Precision
    Precision = float(True_pos_sum) / (Mask_pos_sum + 1e-6)

    
    return Precision


def accuracy(pred_img,label_img):
    p = pred_img.astype(np.float64)
    l = label_img.astype(np.float64)
    if p.max() > 127:
        p /= 255.
    if l.max() > 127:
        l /= 255.

    p = np.clip(p, 0, 1.0)
    l = np.clip(l, 0, 1.0)
    p[p > 0.5] = 1.0
    p[p < 0.5] = 0.0
    l[l > 0.5] = 1.0
    l[l < 0.5] = 0.0
    
#统计ground truth中正样本的个数
    GT_pos_sum = np.sum(l == 1)
    GT_fal_sum = np.sum(l == 0)
#统计预测的mask中正样本的个数
    Mask_pos_sum = np.sum(p == 1)
#统计在groundtruth和mask相同位置都是正样本的个数，即实际为正样本，预测也是正样本的个数
    True_pos_sum = np.sum((l == 1) * (p == 1))
    
#统计在groundtruth和mask相同位置都是fu样本的个数，即实际为fu样本，预测是负样本的个数   
    True_neg_sum = np.sum((l == 0) * (p == 0))
    
    FP = np.sum((l == 0) * (p == 1))
    FT = np.sum((l == 1) * (p == 0))
    
#那么实际为正样本，预测也为正样本占预测的mask中正样本的比例就是Precision
    Acc1 = float((True_pos_sum + True_neg_sum) / (True_pos_sum+True_neg_sum+FP+FT))

    
    return Acc1

    


def calculate_metric(gt, pred): 
    pred[pred>0.5]=1
    pred[pred<1]=0
    confusion = confusion_matrix(gt,pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print('Accuracy:',(TP+TN)/float(TP+TN+FP+FN))
    print('Sensitivity:',TP / float(TP+FN))
    print('Specificity:',TN / float(TN+FP)) 
    





    
    
""" Global metrics"""
    

num = 1000
I = []
Dice = []
J=[]
K = []
assd = []
msd = []
rmse = []
recal = []
precision = []
acc = []

dice = np.zeros(num)


path = "C:/Users/Ming Gong/Desktop/..."
#path = "D:/assemble/p9/"
#path = "C:/Users/Gong Ming/Desktop/unet-master/data/refine/sss/mask/"

files = os.listdir(path) #得到文件夹下的所有文件名称
#files.sort(key=lambda x:int(x[:-4]))

#files = sorted(glob.glob(path + "*.png"), key=lambda x:int(x[:-4]))
images = np.zeros([num,512,512,1])


path1 = "C:/Users/Ming Gong/Desktop/..."
#files1 = sorted(glob.glob(path1 + "*.png"), key=lambda x:int(x[:-4]))
files1= os.listdir(path1) #得到文件夹下的所有文件名称
#files1.sort(key=lambda x:int(x[:-4]))
images1 = np.zeros([num,512,512,1])

for j in range(num):
    
    curr = cv2.imread(path + '/' + files[j],0)
    curr = curr.reshape((1,512,512,1))
    images[j,:,:,:] = curr    
    curr1 = cv2.imread(path1 + '/' + files1[j], 0)
    curr1 = curr1.reshape((1,512,512,1))
    images1[j,:,:,:] = curr1



for l in range(num):
        
    cur = images[l,:,:,:].astype('float32')
    cur1 = images1[l,:,:,:].astype('float32')
        
    rmse.append(Rmse(cur1,cur))
    
    if (cur.sum()<1) or (cur1.sum()<1):
        continue
        
    assd.append(binary_assd(cur1,cur,spacing=None))
    msd.append(binary_hausdorff95(cur1,cur,spacing=None))





I.append(compute_dice(images1,images))
J.append(get_jaccard_index(images1,images))
K.append(compute_rvd(images1,images))
recal.append(recall(images1,images))
precision.append(percision(images1,images))
acc.append(accuracy(images1,images))



newnums = []
for i in assd:
    if 0<i<=5:
        newnums.append(i)
    
asd = np.average(newnums) 

newnums1 = []
for i in msd:
    if 0<i<=30:
        newnums1.append(i)
 

amsd = np.average(newnums1)
