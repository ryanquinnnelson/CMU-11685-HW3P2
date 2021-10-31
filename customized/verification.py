"""
Runs the verification process for face verification.
"""
__author__ = 'ryanquinnnelson'

import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
import pandas as pd
from datetime import datetime
import logging
import torch


class Verification:
    """
    Defines object to manage Face Verification step.
    """

    def __init__(self, data_dir, output_dir, run_name):
        """

        Args:
            data_dir (str): fully qualified path to data directory
            output_dir (str): fully qualified path to output directory
            run_name (str): Name of this run
        """
        logging.info('Initializing verification...')
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.run_name = run_name

    def verify(self, model, epoch, devicehandler):
        """
        Run verification process. For each pair of filenames in the verification_pairs_test.txt, obtain the feature
        embedding using a pretrained model. Compare the feature embeddings using cosine similarity and save the results.
        Format and output all results.

        Args:
            model (nn.Module): pretrained model
            epoch (int): epoch for which model was last trained
            devicehandler (DeviceHandler): object to manage interaction with torch.device

        Returns: None

        """
        logging.info('Performing verification...')
        with open(os.path.join(self.data_dir, 'verification_pairs_test.txt'), 'r') as fd:
            content = fd.readlines()

        composition = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.229, 0.224, 0.225), std=(0.485, 0.456, 0.406))])
        compute_sim = nn.CosineSimilarity(dim=0)

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # for each pair of images in the test file
            results = []
            for i, line in enumerate(content):
                if i % 1000 == 0:
                    logging.info(f'Line {i}...')
                file_a, file_b = line.strip().split()

                # read in images
                img_a = Image.open(os.path.join(self.data_dir, 'verification_data', file_a))
                img_a = composition(img_a).unsqueeze(0)

                img_b = Image.open(os.path.join(self.data_dir, 'verification_data', file_b))
                img_b = composition(img_b).unsqueeze(0)

                if devicehandler.device.type == 'cuda':
                    # move to device
                    img_a = img_a.to(torch.device('cuda'))
                    img_b = img_b.to(torch.device('cuda'))

                # send each image through model and get embedding
                embedding_a, out_a = model.forward(img_a, return_embedding=True)
                embedding_b, out_b = model.forward(img_b, return_embedding=True)

                # calculate similarity
                feats_a = embedding_a.squeeze(0)
                feats_b = embedding_b.squeeze(0)
                sim = compute_sim(feats_a, feats_b)

                # store entry
                results.append([file_a + " " + file_b, sim.item()])

        # build dataframe of the results
        df = pd.DataFrame(results, columns=['Id', 'Category'])

        # save results to file
        filename = f'{self.run_name}.epoch{epoch}.{datetime.now().strftime("%Y%m%d.%H.%M.%S")}.similarity.csv'
        df.to_csv(os.path.join(self.output_dir, filename), header=True, index=False)
