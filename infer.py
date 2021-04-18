"""
Runs an existing traffic analyzer.
"""

from argparse import ArgumentParser
from configparser import ConfigParser
from os.path import exists

from netgen import NetGen
from netgen.backends import CSVBackEnd
from netgen.backends import UIBackEnd

# Parses the input arguments.
parser = ArgumentParser(description="Runs a traffic analyzer.")
parser.add_argument("--quiet", action="store_true", help="disables the logs")
parser.add_argument("--show", help="shows only the flow with these (comma separated) classes")
parser.add_argument("--config", default="netgen_time.conf", help="the name of the configuration file")
parser.add_argument("--model", default="model.joblib", help="the name of the generated model file")
parser.add_argument("--back-end", default="csv", choices=("csv", "ui"), help="the back-end to use")
parser.add_argument("target", help="the name of the pcap file to analyze or the interface to sniff")
args = parser.parse_args()

# Parses the configuration file.
configuration = ConfigParser()
configuration.read(args.config)

netgen = NetGen(configuration, False)
is_pcap = exists(args.target)
if args.back_end == "ui":
    back_end = UIBackEnd(netgen.get_classes(args.model))
else:
    back_end = CSVBackEnd()
if args.show is None:
    classes = []
else:
    classes = args.show.split(",")

while True:
    try:
        results = netgen.infer(args.model, args.target)
        if len(results) > 0 and len(classes) > 0:
            results = results[results["inferred"].isin(classes)]
        back_end.report(results)
        if is_pcap:
            break
    except KeyboardInterrupt:
        break

del back_end
