#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

from protoseg import Augmentation
from protoseg import Config
from protoseg import DataLoader
from protoseg import Model
from protoseg import Predictor
from protoseg import Trainer
from protoseg import backends

datapath = 'data/'
resultspath = 'results/'

backends.register_backend('Cnet_backend', 'Cnet.Cnet_backend.Cnet_backend')
backends.set_backend('Cnet_backend')

# source: https://github.com/EdwardTyantov/ultrasound-nerve-segmentation/blob/master/submission.py
def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return res  # ' '.join([str(r) for r in res])


def help():
    return "Config file parameter missing. Run like: python submit.py /path/to/config.yml"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(help())
        sys.exit(1)

    configs = Config(sys.argv[1])
    for run in configs:
        print("create submission for: ", run)
        resultpath = os.path.join(resultspath, run)
        if not os.path.exists(resultpath):
            os.makedirs(resultpath)
        submissionfile = os.path.join(resultpath, 'submission.csv')
        if os.path.exists(submissionfile):
            os.remove(submissionfile)

        # get config for current run
        config = configs.get()

        backends.set_backend(config['backend'])
        # summary
        summarywriter = backends.backend().get_summary_writer(logdir=resultpath)
        # Load Model
        modelfile = os.path.join('results/', run, 'model.checkpoint')
        model = Model(config, modelfile)
        # dataloader
        dataloader = DataLoader(config=config, mode='test')
        # predictor
        print(backends.backend())
        predictor = Predictor(model=model, config=config, backend=backends.backend())

        with open(submissionfile, 'a') as f:
            f.write('img,pixels\n')
            for index, (img, filename) in tqdm(enumerate(dataloader), total=len(dataloader)):
                mask = predictor.predict(img)
                mask = cv2.resize(
                    mask, (config['orig_width'], config['orig_height']), interpolation=cv2.INTER_NEAREST)
                save = False
                if save:
                    imgfile = os.path.join(resultpath, os.path.basename(
                        filename).split(".")[0]+".png")
                    cv2.imwrite(imgfile, mask*255)
                enc = run_length_enc(mask)
                f.write('{},{}\n'.format(os.path.basename(
                    filename).split(".")[0], ' '.join(map(str, enc))))
    sys.exit(0)
