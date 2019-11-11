import os
import time
import numpy as np
import tensorflow as tf
import pickle
from multiprocessing.pool import ThreadPool

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]

        if self.FLAGS.summary:
            fetches.append(self.summary_op)

        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        if self.FLAGS.summary:
            self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)

def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp : this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo

import math

def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(self.framework.preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)

        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
            self.framework.postprocess(
               prediction, os.path.join(inp_path, this_batch[i])))(*p),
            enumerate(out))
        stop = time.time(); last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

import cv2
from .mAP import define_roc 
from .mAP import read_xml
from .mAP import box_iou

def test(self):
    recall = define_roc(self.FLAGS.labels, list())
    precision = define_roc(self.FLAGS.labels, list())
    TP = define_roc(self.FLAGS.labels, 0)
    FP = define_roc(self.FLAGS.labels, 0)
    FN = define_roc(self.FLAGS.labels, 0)
    for root, dirs, files, in os.walk(self.FLAGS.testXml):
        for file in files:
            ground_truth = read_xml(root+'/'+file)
            path = self.FLAGS.testImg+ground_truth['identify']+'.jpg'
            image = cv2.imread(path)
            results = self.return_predict(image)
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
                            #print(ax,ay,aw,ah,bx,by,bw,bh,iou)
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
                if predict == False:
                    FN[box['name']] += 1
                    r = 0 if TP[box['name']] + FP[box['name']]==0 else TP[box['name']] / (TP[box['name']] + FP[box['name']])
                    p = 0 if TP[box['name']] + FN[box['name']]==0 else TP[box['name']] / (TP[box['name']] + FN[box['name']])
                    recall[box['name']].append(r)
                    precision[box['name']].append(p)
    ap = {'reader box': np.mean(precision['reader box']), 'entry box':np.mean(precision['entry box'])}
    mAP = (ap['reader box'] + ap['entry box'])/2
    return mAP
