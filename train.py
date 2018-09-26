#!/usr/bin/env python

import os
import sys

from protoseg import Augmentation
from protoseg import Config
from protoseg import DataLoader
from protoseg import Model
from protoseg import Trainer
from protoseg import Report
from protoseg import backends

resultspath = 'results/'

def help():
    return "Config file parameter missing. Run like: python train.py /path/to/config.yml"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(help())
        sys.exit(1)

    backends.register_backend('Cnet_backend', 'Cnet.Cnet_backend.Cnet_backend')
    configs = Config(sys.argv[1])    
    for run in configs:
        print("Run: ", run)
        resultpath = os.path.join(resultspath, run)
        if not os.path.exists(resultpath):
            os.makedirs(resultpath)
        configs.save(resultpath + '/config.yml')
        # get config for current run
        config = configs.get()
        # set backend
        backends.set_backend(config['backend'])
        # summary
        summarywriter = backends.backend().get_summary_writer(logdir=resultpath)
        # Load Model
        modelfile = os.path.join('results/', run, 'model.checkpoint')
        model = Model(config, modelfile)
        # augmentation
        augmentation = Augmentation(config=config)
        # data loader
        dataloader = DataLoader(config=config, mode='train', augmentation=augmentation)
        # validation data loader
        valdataloader = DataLoader(config=config, mode='val')
        trainer = Trainer(config, model, dataloader, valdataloader=valdataloader, summarywriter=summarywriter)
        trainer.train()
    
    report = Report(configs, resultspath)
    report.generate()
    
    sys.exit(0)