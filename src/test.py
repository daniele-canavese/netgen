"""
A test file.
"""
import configparser
from collections.abc import Sequence
import socket

import struct
from pandas import DataFrame

from net import TstatAnalyzer
import os
import signal
import sys


def signal_handler(sig, frame):
    print("Closing tstat...")
    analyzer.stop_sniff()


def handle_dataframes(dataframes: Sequence[DataFrame]):
    i = 1
    complete = 0
    nocomplete = 0
    for a in dataframes:
        print("Dataframe " + str(i))
        print("Socket " + socket.inet_ntoa(struct.pack("<L", a['c2s_ip'].iloc[0])) + ":" + str(
            a['c2s_port'].iloc[0]) + " " + socket.inet_ntoa(struct.pack("<L", a['s2c_ip'].iloc[0])) + ":" + str(
            a['s2c_port'].iloc[0]))
        print(a)  # here we should analyze the dataframe with the ML module
        print('\n')
        i += 1
        if a['nocomplete'].iloc[-1] == 1:
            nocomplete += 1
        else:
            complete += 1
    print("Flows: " + str(complete + nocomplete))
    print("Complete flows: " + str(complete))
    print("Nocomplete flows: " + str(nocomplete))
    pass


test_live = 1
netgen_config = configparser.ConfigParser()
netgen_config.read('netgen/netgen.conf')
# print(netgen_config.sections())
analyzer = TstatAnalyzer(netgen_config.get("Default", "MainDirectory") + "/tstat-3.1.1/tstat-conf/tstat.conf", 10)
if test_live == 0:
    os.system("wget https://s3.amazonaws.com/tcpreplay-pcap-files/smallFlows.pcap")
    pcapfile = os.getcwd() + "/smallFlows.pcap"
    result = analyzer.analyze(pcapfile)
    handle_dataframes(result)
    os.system("rm " + pcapfile)
    # Should get at the end of input:
    # Flows: 391
    # Complete flows: 306
    # Nocomplete flows: 85
else:
    # signal.signal(signal.SIGINT, signal_handler)
    while True:  # live capture is supposed to go on indefinitely
        result = analyzer.sniff("eth0")
        handle_dataframes(result)
