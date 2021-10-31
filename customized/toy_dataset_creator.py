"""
Thins out training dataset to create toy dataset for development purposes only.
"""
__author__ = 'ryanquinnnelson'

import os
import sys
import glob


def main():
    dir = sys.argv[1]
    min_dir = None
    min_size = 1000
    dirnames = glob.glob(os.path.join(dir, '*'))

    # calculate minimum set size for image classes
    for d in dirnames:

        dirlength = len(glob.glob(os.path.join(d,'*.jpg')))
        if dirlength < min_size:
            min_dir = d
            min_size = dirlength

    print('min_dir', min_dir)
    print('min_size', min_size)

    # delete all but min_size number of images from directory
    for d in dirnames:
        ld = glob.glob(os.path.join(d,'*.jpg'))
        for i in range(min_size):
            ld.pop()

        for f in ld:
            os.remove(os.path.join(d, f))


if __name__ == "__main__":
    # execute only if run as a script
    main()
