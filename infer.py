"""
Runs an existing traffic analyzer.
"""

from argparse import ArgumentParser
from configparser import ConfigParser
from configparser import MissingSectionHeaderError
from os.path import exists

from netgen import NetGen
from netgen.backends import CSVBackEnd
from netgen.backends import MISPBackEnd
from netgen.backends import UIBackEnd
# Parses the input arguments.
from netgen.misp import MISPEvent
from netgen.misp import MISPServer

parser = ArgumentParser(description="Runs a traffic analyzer.")
parser.add_argument("--quiet", action="store_true", help="disables the logs")
parser.add_argument("--show", help="shows only the flow with these (comma separated) classes")
parser.add_argument("--config", default="netgen.conf", help="the name of the configuration file")
parser.add_argument("--model", default="model.joblib.xz", help="the name of the generated model file")
parser.add_argument("--back-end", default="csv", choices=("csv", "misp", "ui"), help="the back-end to use")
parser.add_argument("target", help="the name of the pcap file, MISP configuration or the interface to use")
args = parser.parse_args()

# Parses the configuration file.
configuration = ConfigParser()
configuration.read(args.config)

netgen = NetGen(configuration, False)

server = None
try:
    misp_configuration = ConfigParser()
    misp_configuration.read(args.target)
    server = MISPServer(misp_configuration, not args.quiet)
    items = server.get_events()
except (ValueError, MissingSectionHeaderError):
    if exists(args.target):
        items = [args.target]
    else:
        items = None

if args.back_end == "ui":
    back_end = UIBackEnd(netgen.get_classes(args.model))
elif args.back_end == "misp":
    back_end = MISPBackEnd(server, len(configuration.get("data_set", "id_fields").split()))
else:
    back_end = CSVBackEnd()
if args.show is None:
    classes = []
else:
    classes = args.show.split(",")

for item in items:
    try:
        if isinstance(item, MISPEvent):  # This is MISP stuff.
            target = item.pcap
        else:
            target = item

        results = netgen.infer(args.model, target)
        if len(results) > 0 and len(classes) > 0:
            results = results[results["inferred"].isin(classes)]

        back_end.report(results, item)
    except KeyboardInterrupt:
        break

del back_end
