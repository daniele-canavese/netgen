"""
Trains a new IDS.
"""

from argparse import ArgumentParser
from configparser import ConfigParser

from colorama import Fore
from colorama import Style
from joblib import dump

from netgen import NetGen

# Parses the input arguments.
parser = ArgumentParser(description="Trains a new IDS.")
parser.add_argument("--quiet", action="store_true", help="disables the logs")
parser.add_argument("--test", default="test", help="sets the test folder")
parser.add_argument("--config", default="netgen.conf", help="the name of the configuration file")
parser.add_argument("--model", default="model.joblib", help="the name of the generated model file")
args = parser.parse_args()

# Parses the configuration file.
configuration = ConfigParser()
configuration.read(args.config)

netgen = NetGen(configuration)
model, train_x, test_x, train_y, test_y = netgen.train(not args.quiet)
if not args.quiet:
    print(Fore.LIGHTYELLOW_EX + "saving the model to %s..." % args.model + Style.RESET_ALL)
dump(model, args.model)
netgen.test(model, train_x, test_x, train_y, test_y, args.test, not args.quiet)
