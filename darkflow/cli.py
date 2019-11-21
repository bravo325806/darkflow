from .defaults import argHandler #Import the default arguments
import os
import zipfile
import time
from .net.build import TFNet
from shutil import copyfile

def cliHandler(args):
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)
    
    requiredDirectories = [FLAGS.imgdir, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.imgdir,'out')]
    if FLAGS.summary:
        requiredDirectories.append(FLAGS.summary)

    _get_dir(requiredDirectories)

    # fix FLAGS.load to appropriate type
    try: FLAGS.load = int(FLAGS.load)
    except: pass
    tfnet = TFNet(FLAGS)
    print(FLAGS.load) 
    if FLAGS.demo:
        tfnet.camera()
        exit('Demo stopped, exit.')

    if FLAGS.train:
        print('Enter training ...'); tfnet.train()
        if not FLAGS.savepb: 
            print('Training finished, exit.')

    if FLAGS.savepb:
        print('Rebuild a constant version ...')
        tfnet.savepb()
        print('save Done')

    if FLAGS.test:
        try: 
            mAP = tfnet.test()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if mAP > FLAGS.testThreshold:
                files = os.listdir('built_graph')
                for file in files:
                    file_type = file.split('.')[-1]
                    os.rename('built_graph/'+file, 'built_graph/model.'+file_type)
                copyfile('parameter/labels.txt', 'built_graph/labels.txt')
                name = FLAGS.model.split('/')[-1][:-4] + '-'+ timestamp + '.zip'
                z = zipfile.ZipFile('built_graph/'+ name, 'w', zipfile.ZIP_DEFLATED)
                z.write('built_graph/model.meta')
                z.write('built_graph/model.pb')
                z.write('built_graph/labels.txt')
                z.close()
                os.remove('built_graph/model.meta')
                os.remove('built_graph/model.pb')
                os.remove('built_graph/labels.txt')
                print('Test correct', 'mAP: ', mAP)
            else:    
                print('Test Fail', 'mAP: ', mAP)
            print('Done')
        finally:
            os.rename('nohup.out', 'parameter/logs/'+timestamp+'.txt')
#tfnet.predict()
