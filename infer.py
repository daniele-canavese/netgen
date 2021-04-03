"""
Runs an existing IDS.
"""

from argparse import ArgumentParser
from configparser import ConfigParser
from os.path import exists

from blessed import Terminal

from netgen import NetGen

# Parses the input arguments.
parser = ArgumentParser(description="Runs an IDS.")
parser.add_argument("--quiet", action="store_true", help="disables the logs")
parser.add_argument("--show", help="shows only the flow with these (comma separated) classes")
parser.add_argument("--config", default="netgen.conf", help="the name of the configuration file")
parser.add_argument("--model", default="model.joblib", help="the name of the generated model file")
parser.add_argument("--ui", default=False, action="store_true", help="uses a very nice UI")
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

terminal = Terminal()
rows = []
labels = {}
if args.ui:
    colors = [terminal.red, terminal.orange, terminal.yellow, terminal.cyan, terminal.blue, terminal.purple,
              terminal.magenta]
    print(terminal.clear() + terminal.home() + terminal.move_xy(0, 0))
    for index, label in enumerate(netgen.get_classes(args.model)):
        labels[label] = colors[index % len(colors)] + label + terminal.normal

while True:
    results = netgen.infer(args.model, args.target)
    if len(results) > 0 and classes is not None:
        results = results[results["inferred"].isin(classes)]
    if len(results) > 0:
        if args.ui:
            index = results.columns.to_list().index("inferred")
            if first:
                print(terminal.move_xy(0, 0) +
                      terminal.bold("  ".join(["%15s" % i for i in results.columns[0:index + 2]])))
                first = False

            for _, row in results.iterrows():
                identifier = " ".join(row[0:index].astype(str))
                inferred = row[index]
                if identifier in rows:
                    line = rows.index(identifier) + 1
                else:
                    line = len(rows) + 1
                    rows.append(identifier)
                    if line >= terminal.height - 1:
                        line -= 1
                        rows.pop(0)
                r = [
                        *["%15s" % i for i in row[0:index]],
                        "%15s" % labels[row["inferred"]],
                        "%15.3f" % row["probability"]
                ]
                print(terminal.move_xy(0, line) + "  ".join(r), end="")
        else:
            if first:
                print(results.to_csv(index=False), end="", flush=True)
                first = False
            else:
                print(results.to_csv(index=False, header=False), end="", flush=True)

    if is_pcap:
        break
