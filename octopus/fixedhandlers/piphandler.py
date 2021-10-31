"""
All things related to installing modules via pip.
"""
__author__ = 'ryanquinnnelson'

import logging
import subprocess


class PipHandler:
    def __init__(self, packages_list):
        logging.info('Initializing pip installations handler...')
        self.packages_list = packages_list if packages_list is not None else []

    def install_packages(self):
        for m in self.packages_list:

            if m == 'ctcdecode':
                _install_ctcdecode()
            else:
                _install_package(m)


def _install_ctcdecode():
    logging.info(f'Downloading ctcdecode...')
    process = subprocess.Popen(['cd'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))

    process = subprocess.Popen(['git', 'clone', '--recursive', 'https://github.com/parlance/ctcdecode.git'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))

    logging.info(f'Installing ctcdecode...')
    process = subprocess.Popen(['cd', 'ctcdecode'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))

    process = subprocess.Popen(['pip', 'install', '.'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))


def _install_package(package):
    # split package into individual words if necessary
    words = package.strip().split()
    commands = ['pip', 'install'] + words
    logging.info(''.join([c + ' ' for c in commands]))
    process = subprocess.Popen(commands,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))
