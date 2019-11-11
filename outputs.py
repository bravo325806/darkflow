from darkflow.net.build import TFNet
import cv2
import numpy as np
import os

def show_rectangle(frame, labels, colors, results, path):
    for result in results:
        label = '{}: {:.0f}%'.format(result['label'], result['confidence'] * 100)
        cv2.rectangle(frame, (result['topleft']['x'],result['topleft']['y']), (result['bottomright']['x'],result['bottomright']['y']), colors[labels.index(result['label'])], 4)
        cv2.putText(frame, label, (result['topleft']['x'],result['topleft']['y']), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 4)
    cv2.imwrite(path, frame)

def load_labels():
    labels = []
    with open('labels.txt') as f:
        for i in f.readlines():
            labels.append(i.replace("\n", ""))
    return labels

def output(labels, colors, options, path):
    tfnet = TFNet(options)
    for root, dirs, files in os.walk('/home/ubuntu/project/object_detection/council/image'):
        for file in files:
            imgcv = cv2.imread(root+'/'+file)
            result = tfnet.return_predict(imgcv)
            show_rectangle(imgcv, labels, colors, result, path+file)

if __name__ == '__main__':
    colors = [tuple(255 * np.random.rand(3)) for i in range(20)]
    labels = load_labels()
    options = {"pbLoad": "built_graph/tiny-yolo-box.pb", "metaLoad": "built_graph/tiny-yolo-box.meta", "threshold": 0.1, "gpu":1.0}
    output(labels, colors, options, '/home/ubuntu/project/object_detection/outputs/box/')
    options = {"pbLoad": "built_graph/tiny-yolo-box2.pb", "metaLoad": "built_graph/tiny-yolo-box2.meta", "threshold": 0.1, "gpu":1.0}
    output(labels, colors, options, '/home/ubuntu/project/object_detection/outputs/box2/')
    options = {"pbLoad": "built_graph/tiny-yolo-box3.pb", "metaLoad": "built_graph/tiny-yolo-box3.meta", "threshold": 0.1, "gpu":1.0}
    output(labels, colors, options, '/home/ubuntu/project/object_detection/outputs/box3/')
