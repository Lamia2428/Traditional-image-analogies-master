from glob import glob
import argparse
from mainanalogy import do_analogy
import os


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./output', help='path to source video')
parser.add_argument('--output', type=str, default='./output/Bp', help='output folder')
args = parser.parse_args()


if not os.path.exists(args.output):
    os.makedirs(args.output)

do_analogy(args.input, args.output)
