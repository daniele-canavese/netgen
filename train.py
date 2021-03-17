"""
Trains a new IDS.
"""

# Parses the input arguments.
from argparse import ArgumentParser
from configparser import ConfigParser
from glob import glob

from colorama import Fore
from colorama import Style
from numpy import mean

from net import TstatAnalyzer

parser = ArgumentParser(description="Trains a new IDS.")
parser.add_argument("config", help="sets the name of the configuration file")
args = parser.parse_args()

# Parses the configuration file.
config = ConfigParser()
config.read(args.config)

print(Fore.RED + "training started" + Style.RESET_ALL)
for name in config["classes"]:
    print(Fore.GREEN + "class " + name + Style.RESET_ALL)
    data = []
    for pcap in sorted(glob(config.get("classes", name), recursive=True)):
        print("  analyzing " + pcap + "...", end="")
        analyzer = TstatAnalyzer(config, 10)
        t = analyzer.analyze(pcap)
        print(" %d samples with an average length of %.1f chunks" % (len(t), mean([len(i) for i in t])))
        data.extend(t)
    print("  %d total samples with an average length of %.1f chunks" % (len(data), mean([len(i) for i in data])))
