"""
A test file.
"""

from net import TstatAnalyzer

analyzer = TstatAnalyzer("tstat.conf", 10)
print(analyzer.analyze("file.pcap"))
print(analyzer.sniff("eth0"))
