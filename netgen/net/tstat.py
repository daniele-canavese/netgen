"""
Tstat class.
"""
from collections import Sequence
from configparser import ConfigParser
from ctypes import CDLL
from ctypes import POINTER
from ctypes import Structure
from ctypes import c_char_p
from ctypes import c_double
from ctypes import c_int
from ctypes import c_ubyte
from ctypes import c_uint
from ctypes import c_uint32
from ctypes import c_ulong
from ctypes import c_ushort
from socket import inet_ntoa
from struct import pack
from typing import Any
from typing import Dict

from pandas import DataFrame

from .analysis import Analyzer


class TstatAnalyzer(Analyzer):
    """
    The tstat analyzer.
    """

    def __init__(self, configuration: ConfigParser, packets: int) -> None:
        """
        Creates the analyzer.

        :param configuration: the netgen configuration
        :param packets: the number of packets after which the analysis functions will return
        """

        self.__packets = packets
        self.__configuration = configuration
        # noinspection SpellCheckingInspection
        self.__dataframe_columns = {
                "c2s_ip":                 "category",
                "c2s_port":               "uint16",
                "c2s_packets":            "uint32",
                "c2s_reset_count":        "uint32",
                "c2s_ack_pkts":           "uint32",
                "c2s_pureack_pkts":       "uint32",
                "c2s_unique_bytes":       "uint64",
                "c2s_data_pkts":          "uint32",
                "c2s_data_bytes":         "uint64",
                "c2s_rexmit_pkts":        "uint32",
                "c2s_rexmit_bytes":       "uint64",
                "c2s_out_order_pkts":     "uint32",
                "c2s_syn_count":          "uint32",
                "c2s_fin_count":          "uint32",
                "s2c_ip":                 "category",
                "s2c_port":               "uint16",
                "s2c_packets":            "uint32",
                "s2c_reset_count":        "uint32",
                "s2c_ack_pkts":           "uint32",
                "s2c_pureack_pkts":       "uint32",
                "s2c_unique_bytes":       "uint64",
                "s2c_data_pkts":          "uint32",
                "s2c_data_bytes":         "uint64",
                "s2c_rexmit_pkts":        "uint32",
                "s2c_rexmit_bytes":       "uint64",
                "s2c_out_order_pkts":     "uint32",
                "s2c_syn_count":          "uint32",
                "s2c_fin_count":          "uint32",
                "first_time":             "float32",
                "last_time":              "float32",
                "completion_time":        "float32",
                "c2s_payload_start_time": "float32",
                "c2s_payload_end_time":   "float32",
                "c2s_ack_start_time":     "float32",
                "s2c_payload_start_time": "float32",
                "s2c_payload_end_time":   "float32",
                "s2c_ack_start_time":     "float32",
                "complete":               "bool",
                "reset":                  "bool",
                "nocomplete":             "bool"
        }

        # noinspection SpellCheckingInspection
        self.__libtstat = CDLL(self.__configuration.get("tstat", "library"))
        self.__libtstat.tstat_export_core_statistics_init.restype = c_int
        # noinspection SpellCheckingInspection
        self.__libtstat.tstat_export_core_statistics_init.argtypes = [c_char_p, c_char_p]
        self.__conf_file = c_char_p(self.__configuration.get("tstat", "configuration").encode("utf-8"))
        self.__sniffing_initialized = False
        self.__sniffing_last_dictionary = {}

    def analyze(self, file: str) -> Sequence[DataFrame]:
        """
        Analyzes a capture file.

        :param file: the file name to analyze
        :return: a list of dataframes where each dataframe contains the time steps of a flow
        """

        dictionary = {}

        pcap_file = c_char_p(file.encode("utf-8"))
        self.__libtstat.tstat_export_core_statistics_init(self.__conf_file, pcap_file, 0, 0, 0)

        result = True
        while result:
            result = self.__read_tstat_chunk(dictionary)

        self.__libtstat.tstat_export_core_statistics_close(0)

        for flow_id in dictionary.keys():
            d = DataFrame(columns=self.__dataframe_columns.keys(), data=dictionary[flow_id])
            for column, kind in self.__dataframe_columns.items():
                d[column] = d[column].astype(kind)
            dictionary[flow_id] = d

        return list(dictionary.values())

    def __read_tstat_chunk(self, dictionary: Dict[int, Any]) -> bool:
        """
        Reads a chuck from a capture.

        :param dictionary: the dictionary to fill with the chuck data
        :return: a value stating if the read was successful or not
        """

        res = self.__libtstat.tstat_export_core_statistics_read_chunk(self.__packets, 0) == 1

        core_statistics_list_cursor = POINTER(TcsListElem)
        core_statistics_list_cursor = core_statistics_list_cursor.in_dll(self.__libtstat, "tcs_list_start")

        core_statistics_list_elements = 0
        core_statistics_list_elements_complete = 0
        # noinspection SpellCheckingInspection
        core_statistics_list_elements_nocomplete = 0

        while True:
            if core_statistics_list_cursor:
                core_statistics_list_elements += 1
                core_statistics = core_statistics_list_cursor.contents.stat.contents

                if core_statistics.nocomplete != 0:
                    core_statistics_list_elements_nocomplete += 1
                else:
                    core_statistics_list_elements_complete += 1

                flow_id = core_statistics.id_number
                row = core_statistics.to_dictionary()
                if flow_id not in dictionary:
                    dictionary[flow_id] = [row]
                else:
                    dictionary[flow_id].append(row)

                if core_statistics_list_cursor.contents.next:
                    core_statistics_list_cursor = core_statistics_list_cursor.contents.next
                else:
                    break
            else:
                return res

        self.__libtstat.tstat_tcs_list_release()
        return res

    def sniff(self, interface: str) -> Sequence[DataFrame]:
        """
        Sniffs an interface.

        :param interface: the name of the interface to sniff
        :return: a list of dataframes where each dataframe contains the time steps of a flow
        """

        snapshot_length = self.__configuration.getint("tstat", "snapshot_length")
        timeout = int(self.__configuration.getfloat("tstat", "timeout") * 1000)
        chunks_length = self.__configuration.getint("tstat", "chunks_length")

        dictionary = {}

        if self.__sniffing_initialized is False:
            interface_file = c_char_p(interface.encode("utf-8"))
            self.__libtstat.tstat_export_core_statistics_init(self.__conf_file, interface_file, 1, snapshot_length,
                                                              timeout)
            self.__sniffing_initialized = True

        chunk_number = 1
        while chunk_number - 1 < chunks_length:
            self.__read_tstat_chunk(dictionary)
            chunk_number += 1

        dataframe_dictionary = {}
        for flow_id in dictionary.keys():
            if flow_id in self.__sniffing_last_dictionary.keys():
                last_dictionary_flow_stats = self.__sniffing_last_dictionary[flow_id]
                dictionary[flow_id][0:0] = last_dictionary_flow_stats
            dataframe_dictionary[flow_id] = DataFrame(data=dictionary[flow_id], columns=self.__dataframe_columns)
        self.__sniffing_last_dictionary = dictionary

        return list(dataframe_dictionary.values())

    def __del__(self) -> None:
        """
        Stop sniffing an interface, closing gracefully the underlying Tstat instance.
        """

        if self.__sniffing_initialized:
            self.__libtstat.tstat_export_core_statistics_close(0)
            self.__sniffing_initialized = False


class TCPCoreStatistics(Structure):
    """
    The TCP core statistics of tstat.
    """

    """ The statistics. """
    # noinspection SpellCheckingInspection
    _fields_ = [("id_number", c_ulong),

                ("c2s_ip", c_uint32),
                ("c2s_port", c_ushort),
                ("c2s_packets", c_ulong),
                ("c2s_reset_count", c_ubyte),
                ("c2s_ack_pkts", c_ulong),
                ("c2s_pureack_pkts", c_ulong),
                ("c2s_unique_bytes", c_ulong),
                ("c2s_data_pkts", c_ulong),
                ("c2s_data_bytes", c_ulong),
                ("c2s_rexmit_pkts", c_uint),
                ("c2s_rexmit_bytes", c_uint),
                ("c2s_out_order_pkts", c_uint),
                ("c2s_syn_count", c_ubyte),
                ("c2s_fin_count", c_ubyte),

                ("s2c_ip", c_uint32),
                ("s2c_port", c_ushort),
                ("s2c_packets", c_ulong),
                ("s2c_reset_count", c_ubyte),
                ("s2c_ack_pkts", c_ulong),
                ("s2c_pureack_pkts", c_ulong),
                ("s2c_unique_bytes", c_ulong),
                ("s2c_data_pkts", c_ulong),
                ("s2c_data_bytes", c_ulong),
                ("s2c_rexmit_pkts", c_uint),
                ("s2c_rexmit_bytes", c_uint),
                ("s2c_out_order_pkts", c_uint),
                ("s2c_syn_count", c_ubyte),
                ("s2c_fin_count", c_ubyte),

                ("first_time", c_double),
                ("last_time", c_double),
                ("completion_time", c_double),

                ("c2s_payload_start_time", c_double),
                ("c2s_payload_end_time", c_double),
                ("c2s_ack_start_time", c_double),
                ("s2c_payload_start_time", c_double),
                ("s2c_payload_end_time", c_double),
                ("s2c_ack_start_time", c_double),

                ("complete", c_ubyte),
                ("reset", c_ubyte),
                ("nocomplete", c_ubyte)]

    # noinspection DuplicatedCode
    def __str__(self) -> str:
        """
        Converts this statistic to a printable string.

        :return: the string representation of the statistics
        """

        printed_statistics = str(self.id_number) + " "
        printed_statistics += str(inet_ntoa(pack("<L", self.c2s_ip))) + " "
        printed_statistics += str(self.c2s_port) + " "
        printed_statistics += str(self.c2s_packets) + " "
        printed_statistics += str(self.c2s_reset_count) + " "
        printed_statistics += str(self.c2s_ack_pkts) + " "
        printed_statistics += str(self.c2s_pureack_pkts) + " "
        printed_statistics += str(self.c2s_unique_bytes) + " "
        printed_statistics += str(self.c2s_data_pkts) + " "
        printed_statistics += str(self.c2s_data_bytes) + " "
        printed_statistics += str(self.c2s_rexmit_pkts) + " "
        printed_statistics += str(self.c2s_rexmit_bytes) + " "
        printed_statistics += str(self.c2s_out_order_pkts) + " "
        printed_statistics += str(self.c2s_syn_count) + " "
        printed_statistics += str(self.c2s_fin_count) + " "
        printed_statistics += str(inet_ntoa(pack("<L", self.s2c_ip))) + " "
        printed_statistics += str(self.s2c_port) + " "
        printed_statistics += str(self.s2c_packets) + " "
        printed_statistics += str(self.s2c_reset_count) + " "
        printed_statistics += str(self.s2c_ack_pkts) + " "
        printed_statistics += str(self.s2c_pureack_pkts) + " "
        printed_statistics += str(self.s2c_unique_bytes) + " "
        printed_statistics += str(self.s2c_data_pkts) + " "
        printed_statistics += str(self.s2c_data_bytes) + " "
        printed_statistics += str(self.s2c_rexmit_pkts) + " "
        printed_statistics += str(self.s2c_rexmit_bytes) + " "
        printed_statistics += str(self.s2c_out_order_pkts) + " "
        printed_statistics += str(self.s2c_syn_count) + " "
        printed_statistics += str(self.s2c_fin_count) + " "
        printed_statistics += str(self.first_time / 1000) + " "
        printed_statistics += str(self.last_time / 1000) + " "
        printed_statistics += str(self.completion_time / 1000) + " "
        printed_statistics += str(self.c2s_payload_start_time / 1000) + " "
        printed_statistics += str(self.c2s_payload_end_time / 1000) + " "
        printed_statistics += str(self.c2s_ack_start_time / 1000) + " "
        printed_statistics += str(self.s2c_payload_start_time / 1000) + " "
        printed_statistics += str(self.s2c_payload_end_time / 1000) + " "
        printed_statistics += str(self.s2c_ack_start_time / 1000) + " "
        printed_statistics += str(self.complete) + " "
        printed_statistics += str(self.reset) + " "
        printed_statistics += str(self.nocomplete)
        if self.nocomplete != 0:
            # noinspection SpellCheckingInspection
            printed_statistics += " NOCOMPLETE"
        else:
            printed_statistics += " COMPLETE"

        return printed_statistics

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Returns this statistic as a dictionary.
        
        :return: the dictionary equivalent of the statistics. 
        """

        # noinspection SpellCheckingInspection
        row = {
                "c2s_ip":                 inet_ntoa(pack("<L", self.c2s_ip)),
                "c2s_port":               self.c2s_port,
                "c2s_packets":            self.c2s_packets,
                "c2s_reset_count":        self.c2s_reset_count,
                "c2s_ack_pkts":           self.c2s_ack_pkts,
                "c2s_pureack_pkts":       self.c2s_pureack_pkts,
                "c2s_unique_bytes":       self.c2s_unique_bytes,
                "c2s_data_pkts":          self.c2s_data_pkts,
                "c2s_data_bytes":         self.c2s_data_bytes,
                "c2s_rexmit_pkts":        self.c2s_rexmit_pkts,
                "c2s_rexmit_bytes":       self.c2s_rexmit_bytes,
                "c2s_out_order_pkts":     self.c2s_out_order_pkts,
                "c2s_syn_count":          self.c2s_syn_count,
                "c2s_fin_count":          self.c2s_fin_count,
                "s2c_ip":                 inet_ntoa(pack("<L", self.s2c_ip)),
                "s2c_port":               self.s2c_port,
                "s2c_packets":            self.s2c_packets,
                "s2c_reset_count":        self.s2c_reset_count,
                "s2c_ack_pkts":           self.s2c_ack_pkts,
                "s2c_pureack_pkts":       self.s2c_pureack_pkts,
                "s2c_unique_bytes":       self.s2c_unique_bytes,
                "s2c_data_pkts":          self.s2c_data_pkts,
                "s2c_data_bytes":         self.s2c_data_bytes,
                "s2c_rexmit_pkts":        self.s2c_rexmit_pkts,
                "s2c_rexmit_bytes":       self.s2c_rexmit_bytes,
                "s2c_out_order_pkts":     self.s2c_out_order_pkts,
                "s2c_syn_count":          self.s2c_syn_count,
                "s2c_fin_count":          self.s2c_fin_count,
                "first_time":             self.first_time,
                "last_time":              self.last_time,
                "completion_time":        self.completion_time,
                "c2s_payload_start_time": self.c2s_payload_start_time,
                "c2s_payload_end_time":   self.c2s_payload_end_time,
                "c2s_ack_start_time":     self.c2s_ack_start_time,
                "s2c_payload_start_time": self.s2c_payload_start_time,
                "s2c_payload_end_time":   self.s2c_payload_end_time,
                "s2c_ack_start_time":     self.s2c_ack_start_time,
                "complete":               self.complete,
                "reset":                  self.reset,
                "nocomplete":             self.nocomplete
        }

        return row


class TcsListElem(Structure):
    """
    An element in a double-linked list of TCP core statistics.
    """

    pass


TcsListElem._fields_ = [("next", POINTER(TcsListElem)),
                        ("prev", POINTER(TcsListElem)),
                        ("stat", POINTER(TCPCoreStatistics))]
