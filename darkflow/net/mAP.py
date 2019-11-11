import numpy as np
from bs4 import BeautifulSoup

def read_xml(path):
    content = open(path)
    soup = BeautifulSoup(content, 'html.parser')
    identify = soup.find_all('filename')[0].get_text().split('.')[0]
    img_path = soup.find_all('path')[0].get_text()
    width = soup.find_all('width')[0].get_text()
    height = soup.find_all('height')[0].get_text()
    tmp = list()
    for i in range(len(soup.find_all('name'))):
        bb_name = soup.find_all('name')[i].get_text()
        xmin = soup.find_all('xmin')[i].get_text()
        ymin = soup.find_all('ymin')[i].get_text()
        xmax = soup.find_all('xmax')[i].get_text()
        ymax = soup.find_all('ymax')[i].get_text()
        tmp.append({'name':bb_name, 'xmin':int(xmin), 'ymin':int(ymin), 'xmax':int(xmax), 'ymax':int(ymax)})
    return {'identify': identify, 'path': img_path, 'width':int(width), 'height':int(height), 'object':tmp}

def overlap_c(x1, w1, x2, w2):
    l1 = x1 - w1 /2.
    l2 = x2 - w2 /2.
    left = max(l1,l2)
    r1 = x1 + w1 /2.
    r2 = x2 + w2 /2.
    right = min(r1, r2)
    return right - left

def box_intersection_c(ax, ay, aw, ah, bx, by, bw, bh):
    w = overlap_c(ax, aw, bx, bw)
    h = overlap_c(ay, ah, by, bh)
    if w<0 or h<0: return 0
    area = w *h
    return area

def box_iou(ax, ay, aw, ah, bx, by, bw, bh):
    intersection = box_intersection_c(ax, ay, aw, ah, bx, by, bw, bh)
    union = aw * ah + bw * bh - intersection
    return intersection/union

def load_labels(path):
    labels = []
    with open('labels.txt') as f:
        for i in f.readlines(): # 讀取檔案的每一行資料
            labels.append(i.replace("\n", "")) # 新增類別
    return labels

def define_roc(labels, data_type):
    tmp = {}
    if type(data_type) == list:
        [tmp.update({x:data_type.copy()}) for x in load_labels(labels)]
    else:
        [tmp.update({x:data_type}) for x in load_labels(labels)]
    return tmp
