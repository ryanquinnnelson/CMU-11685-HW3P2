"""
All things related to kaggle.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
import subprocess
import json
import glob
import zipfile

import octopus.helper as utilities


class KaggleConnector:
    """
    Defines an object which manages the connection to kaggle.
    """

    def __init__(self, kaggle_dir, content_dir, token_file, competition, delete_zipfiles):
        """
        Initialize the kaggle connector.
        Args:
            kaggle_dir (str): fully-qualified location where kaggle root directory should be placed
            content_dir (str): fully-qualified directory where content downloaded from kaggle should be placed
            token_file (str):  fully-qualified location of the kaggle api file "kaggle.json" to be copied into the kaggle root directory
            competition (str): name of the kaggle competition from which data should be downloaded
            delete_zipfiles (Boolean): if True, removes any zipfiles downloaded by kaggle after unzipping them
        """
        logging.info('Initializing kaggle connector...')
        self.kaggle_dir = kaggle_dir
        self.content_dir = content_dir
        self.token_file = token_file
        self.competition = competition
        self.delete_zipfiles = delete_zipfiles
        self.competition_dir = os.path.join(content_dir, 'competitions', competition)

    def setup(self):
        """
        Perform all tasks needed to set up kaggle in the local environment. Create kaggle root and
        content directories, copy the "kaggle.json" api token to the kaggle root directory, and configure kaggle
        to use the content directory specified.
        Returns: None

        """
        logging.info('Setting up kaggle connection...')

        # create directories
        _mkdirs(self.kaggle_dir, self.content_dir)

        # setup for kaggle api token
        token = _read_kaggle_token(self.token_file)
        token_dest = _write_kaggle_token(token, self.kaggle_dir)
        _secure_kaggle_token(token_dest)

        # configure kaggle to use content directory
        _configure_content_dir(self.content_dir)

    def download(self):
        """
        Download data associated with the kaggle competition.
        Returns: None
        """

        if not os.path.isdir(self.competition_dir):
            logging.info(f'Downloading files for kaggle competition:{self.competition}...')
            process = subprocess.Popen(['kaggle', 'competitions', 'download', '-c', self.competition],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            logging.info(stdout.decode("utf-8"))
        else:
            logging.info('Competition files are already downloaded.')

    def unzip(self):
        """
        Unzip any .zip files found in the competition subdirectory inside the content directory.
        Returns: None

        """

        logging.info('Unzipping competition files...')
        # get filenames
        zipfiles = glob.glob(self.competition_dir + '/*.zip')
        logging.info(f'Found the following zipped files:{zipfiles}.')
        # unzip each file
        for f in zipfiles:
            with zipfile.ZipFile(f, 'r') as zip_ref:
                zip_ref.extractall(self.competition_dir)

        # clean up original zipfile
        if self.delete_zipfiles:
            logging.info('Removing zip files after extracting contents...')
            for f in zipfiles:
                os.remove(f)

    def download_and_unzip(self):
        """
        Download and unzip kaggle competition data.
        Returns: None

        """

        self.download()
        self.unzip()


def _configure_content_dir(content_dir):
    """
    Configure kaggle to use the supplied content directory.
    Args:
        content_dir (str): fully qualified directory into which kaggle content will be downloaded

    Returns:None

    """

    logging.info('Configuring content directory for kaggle...')
    process = subprocess.Popen(['kaggle', 'config', 'set', '-n', 'path', '-v', content_dir],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))


def _mkdirs(kaggle_dir, content_dir):
    """
    Create kaggle root and content directories, if necessary.
    Args:
        kaggle_dir (str): fully-qualified location of kaggle root directory
        content_dir (str): fully-qualified location of kaggle content

    Returns:None

    """

    logging.info('Setting up kaggle directories...')

    # kaggle directory
    utilities.create_directory(kaggle_dir)

    # kaggle content directory
    utilities.create_directory(content_dir)


def _read_kaggle_token(token_file):
    """
    Read the token file.
    Args:
        token_file (str): fully-qualified filename of the kaggle api token. Expects content to be in json format.

    Returns:json object representing token contents

    """

    logging.info(f'Reading kaggle token from {token_file}...')
    with open(token_file) as token_source:
        token = json.load(token_source)
        return token


def _write_kaggle_token(token, kaggle_dir):
    """
    Create token file in the kaggle directory using supplied token value.
    Args:
        token (json): json object containing value to be written to file.
        kaggle_dir (str): fully-qualified location of kaggle root directory.

    Returns:str representing fully-qualified filename where token file is written

    """

    logging.info(f'Writing kaggle token to {kaggle_dir}...')
    token_dest = os.path.join(kaggle_dir, 'kaggle.json')
    with open(token_dest, 'w') as file:
        json.dump(token, file)

    return token_dest


def _secure_kaggle_token(token_dest):
    """
    Secures the token file so only the current user can access it.
    Args:
        token_dest (str): fully-qualified filename where token file is written

    Returns:None

    """

    logging.info('Securing kaggle token...')
    process = subprocess.Popen(['chmod', '600', token_dest],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))
