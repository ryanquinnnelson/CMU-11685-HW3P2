"""
All things related to installing modules via pip.
"""
__author__ = 'ryanquinnnelson'

import os
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
    # check if it is already installed
    process = subprocess.Popen(['pip', 'show', 'ctcdecode'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    rc = process.returncode
    if rc == 0:
        logging.info('ctcdecode is already installed.')
    else:

        logging.info(f'Downloading ctcdecode...')
        home_dir = os.path.expanduser('~')
        os.chdir(home_dir)

        process = subprocess.Popen(['git', 'clone', '--recursive', 'https://github.com/parlance/ctcdecode.git'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        logging.info(stdout.decode("utf-8"))
        logging.info(stderr.decode("utf-8"))

        # workaround for ctcdecode installation issue
        # Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp-7c85b1e2.so.1 library.
        # Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
        # https://github.com/pytorch/pytorch/issues/37377
        os.environ['MKL_THREADING_LAYER'] = 'GNU'

        logging.info(f'Installing ctcdecode...')
        os.chdir(os.path.join(home_dir, 'ctcdecode'))
        commands = ['pip', 'install', os.path.join(home_dir, 'ctcdecode')]
        logging.info(''.join([c + ' ' for c in commands]))
        process = subprocess.Popen(['pip', 'install', os.path.join(home_dir, 'ctcdecode')],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        logging.info(stdout.decode("utf-8"))
        logging.info(stderr.decode("utf-8"))


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
