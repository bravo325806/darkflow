import sys
import numpy as np
import cv2
import tensorflow as tf
import os
from bs4 import BeautifulSoup
from darkflow.net.build import TFNet

# sigmoid計算
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 轉換圖像
def resize_img(img, reqsize):
    img = cv2.resize(img, (reqsize, reqsize), interpolation=cv2.INTER_AREA) # 將圖像下採樣成416x416
    img = img[:,:,::-1] # 將顏色轉換為rgb
#    img = img.transpose((2, 0 , 1)) # 將維度轉換為channel, height, weight
    #img = img.reshape(1, 3, 416, 416) # 將維度擴展為num, channel, height, weight
    return img/255.0 # 因應float型態，故先除255

# 讀取類別檔
def load_labels():
    labels = []
    with open('labels.txt') as f:
        for i in f.readlines(): # 讀取檔案的每一行資料
            labels.append(i.replace("\n", "")) # 新增類別
    return labels

# 新增樣本
def read_xml(path):
    content = open(root+'/'+file)

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

def img_output(path, image, labels, colors, results):
    img = np.copy(image)    
    if len(results) == 9:
        cv2.rectangle(img, (results['xmin'],results['ymin']), (results['xmax'],results['ymax']), colors[labels.index(results['name'])], 4)
        cv2.putText(img, results['name'], (results['xmin'], results['ymin']), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 4)
    else:
        for result in results:
            label = '{}: {:.0f}%'.format(result['label'], result['confidence'] * 100)
            cv2.rectangle(img, (result['topleft']['x'],result['topleft']['y']), (result['bottomright']['x'],result['bottomright']['y']), colors[labels.index(result['label'])], 4)
            cv2.putText(img, label, (result['topleft']['x'],result['topleft']['y']), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 4)    
    cv2.imwrite(path, img)
if __name__ == '__main__':
    labels = load_labels() # 讀取類別檔
    
    colors = [tuple(255 * np.random.rand(3)) for i in range(20)]
    recall = {'reader box': list(), 'entry box': list()}
    precision = {'reader box':list(), 'entry box': list()}
    TP = {'reader box':0, 'entry box':0}
    FP = {'reader box':0, 'entry box':0}
    FN = {'reader box':0, 'entry box':0}
    options = {"pbLoad": "built_graph/tiny-yolo-box3.pb", "metaLoad": "built_graph/tiny-yolo-box3.meta", "threshold": 0.1, "gpu":1.0}
    tfnet = TFNet(options)
    for root, dirs, files, in os.walk('/home/ubuntu/project/object_detection/council/xml'):
        for file in files:
            ground_truth = read_xml(root+'/'+file)
            path = '/home/ubuntu/project/object_detection/council/image/'+ground_truth['identify']+'.jpg'
            image = cv2.imread(path)
            results = tfnet.return_predict(image)
            for box in ground_truth['object']:
                predict = False
                for result in results:
                    if result['label'] == box['name']:
                        if predict==False:
                            ax = box['xmin'] + int((box['xmax'] - box['xmin']) / 2)
                            ay = box['ymin'] + int((box['ymax'] - box['ymin']) / 2)
                            aw = box['xmax'] - box['xmin']
                            ah = box['ymax'] - box['ymin']
                            bx = result['topleft']['x'] + int((result['bottomright']['x'] - result['topleft']['x']) / 2)
                            by = result['topleft']['y'] + int((result['bottomright']['y'] - result['topleft']['y']) / 2)
                            bw = result['bottomright']['x'] -  result['topleft']['x']
                            bh = result['bottomright']['y'] -  result['topleft']['y']
                            iou = box_iou(ax,ay,aw,ah,bx,by,bw,bh)
                            #print(box['name'], ax,ay,aw,ah,bx,by,bw,bh,iou)
                            if iou >=0.4:
                                TP[box['name']] += 1
                                r = 0 if TP[box['name']] + FP[box['name']]==0 else TP[box['name']] / (TP[box['name']] + FP[box['name']])
                                p = 0 if TP[box['name']] + FN[box['name']]==0 else TP[box['name']] / (TP[box['name']] + FN[box['name']])
                                recall[box['name']].append(r)
                                precision[box['name']].append(p)
                                predict=True
                            else:
                                FP[box['name']] += 1
                                r = 0 if TP[box['name']] + FP[box['name']]==0 else TP[box['name']] / (TP[box['name']] + FP[box['name']])
                                p = 0 if TP[box['name']] + FN[box['name']]==0 else TP[box['name']] / (TP[box['name']] + FN[box['name']])
                                recall[box['name']].append(r)
                                precision[box['name']].append(p)
                if predict ==False:
                    FN[box['name']] += 1
                    r = 0 if TP[box['name']] + FP[box['name']]==0 else TP[box['name']] / (TP[box['name']] + FP[box['name']])
                    p = 0 if TP[box['name']] + FN[box['name']]==0 else TP[box['name']] / (TP[box['name']] + FN[box['name']])
                    recall[box['name']].append(r)
                    precision[box['name']].append(p)

                #img_output('outputs/cpu/'+ground_truth['name']+'_' +ground_truth['identify']+'.jpg', image, labels, colors, boxesInfo)
    ap = {'reader box': np.mean(precision['reader box']), 'entry box':np.mean(precision['entry box'])}
    mAP = (ap['reader box'] + ap['entry box'])/2
    print(ap)
    print(mAP)
    print('Finished')
