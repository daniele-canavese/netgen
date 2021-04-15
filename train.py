"""
Trains a new traffic analyzer.
"""

from argparse import ArgumentParser
from configparser import ConfigParser
from random import seed as random_seed

from blessed import Terminal
from joblib import dump
from numpy.random import seed as numpy_seed
from torch import manual_seed as torch_seed

from netgen import NetGen

# Parses the input arguments.
parser = ArgumentParser(description="Trains a new traffic analyzer.")
parser.add_argument("--quiet", action="store_true", help="disables the logs")
parser.add_argument("--test", default="test", help="the test report folder")
parser.add_argument("--seed", type=int, default=42, help="the seed for the pseudo-random number generators")
parser.add_argument("--config", default="netgen.conf", help="the name of the configuration file")
parser.add_argument("--model", default="model.joblib", help="the name of the generated model file")
parser.add_argument("data", help="the name of the data file")
args = parser.parse_args()

# Parses the configuration file.
configuration = ConfigParser()
configuration.read(args.config)

terminal = Terminal()

random_seed(args.seed)
numpy_seed(args.seed)
torch_seed(args.seed)

# Trains! A lot of trains!
netgen = NetGen(configuration, not args.quiet)
model, train_x, test_x, train_y, test_y = netgen.train(args.data)
dump(model, args.model)
netgen.test(model, train_x, test_x, train_y, test_y, args.test)
