"""
Runs an existing IDS.
"""

from argparse import ArgumentParser
from configparser import ConfigParser
from os.path import exists

from netgen import NetGen

# Parses the input arguments.
parser = ArgumentParser(description="Runs an IDS.")
parser.add_argument("--quiet", action="store_true", help="disables the logs")
parser.add_argument("--show", help="shows only the flow with these (comma separated) classes")
parser.add_argument("--config", default="netgen.conf", help="the name of the configuration file")
parser.add_argument("--model", default="model.joblib", help="the name of the generated model file")
parser.add_argument("target", help="the name of the pcap file to analyze or the interface to sniff")
args = parser.parse_args()

# Parses the configuration file.
configuration = ConfigParser()
configuration.read(args.config)

is_pcap = exists(args.target)
if args.show is None:
    classes = None
else:
    classes = args.show.split(",")
first = True

netgen = NetGen(configuration)
while True:
    results = netgen.infer(args.model, args.target)
    if len(results) > 0 and classes is not None:
        results = results[results["inferred"].isin(classes)]
    if len(results) > 0:
        if first:
            print(results.to_csv(index=False), end="", flush=True)
            first = False
        else:
            print(results.to_csv(index=False, header=False), end="", flush=True)
    if is_pcap:
        break
