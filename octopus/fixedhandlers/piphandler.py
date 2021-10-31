"""
All things related to installing modules via pip.
"""
__author__ = 'ryanquinnnelson'

import logging
import subprocess


class PipHandler:
    def __init__(self, modules):
        self.modules = modules if modules is not None else []

    def install_modules(self):
        for m in self.modules:

            if m == 'ctcdecode':
                install_ctcdecode()
            else:
                install_module(m)


def install_ctcdecode():
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


def install_module(module):
    logging.info(f'Installing {module}...')
    process = subprocess.Popen(['pip', 'install', module],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))
