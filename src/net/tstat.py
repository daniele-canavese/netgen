"""
Tstat class.
"""

from collections import Sequence
from typing import Any, Dict

from pandas import DataFrame

from .tstat_statistics_list_element import TcsListElem
from .analysis import Analyzer

from ctypes import *

import configparser


class TstatAnalyzer(Analyzer):
    """
    The tstat analyzer.
    """

    def __init__(self, configuration: str, packets: int) -> None:
        """
        Creates the analyzer.

        :param configuration: the name of the configuration file for tstat
        :param packets: the number of packets after which the analysis functions will return
        """

        self.__packets = packets
        self.__configuration = configuration

        self.__dataframe_columns = ['c2s_ip', 'c2s_port', 'c2s_packets', 'c2s_reset_count', 'c2s_ack_pkts',
                                    'c2s_pureack_pkts',
                                    'c2s_unique_bytes', 'c2s_data_pkts', 'c2s_data_bytes', 'c2s_rexmit_pkts',
                                    'c2s_rexmit_bytes',
                                    'c2s_out_order_pkts', 'c2s_syn_count', 'c2s_fin_count',
                                    's2c_ip', 's2c_port', 's2c_packets', 's2c_reset_count', 's2c_ack_pkts',
                                    's2c_pureack_pkts',
                                    's2c_unique_bytes', 's2c_data_pkts', 's2c_data_bytes', 's2c_rexmit_pkts',
                                    's2c_rexmit_bytes',
                                    's2c_out_order_pkts', 's2c_syn_count', 's2c_fin_count',
                                    'first_time', 'last_time', 'completion_time', 'c2s_payload_start_time',
                                    'c2s_payload_end_time',
                                    'c2s_ack_start_time', 's2c_payload_start_time', 's2c_payload_end_time',
                                    's2c_ack_start_time',
                                    'complete', 'reset', 'nocomplete']

        self.__netgen_config = configparser.ConfigParser()
        self.__netgen_config.read('netgen/netgen.conf')
        print(self.__netgen_config.sections())
        self.__libtstat = CDLL(self.__netgen_config.get("Default", "MainDirectory") + "/tstat-3.1.1/libtstat/.libs"
                                                                                      "/libtstat.so")
        self.__libtstat.tstat_export_core_statistics_init.restype = c_int
        self.__libtstat.tstat_export_core_statistics_init.argtypes = [c_char_p, c_char_p]
        self.__conf_file = c_char_p(self.__configuration.encode('utf-8'))
        self.__sniffing_initialized = False
        self.__sniffing_last_dictionary = {}

    def read_tstat_chunk(self, dictionary: Dict[int, Any]) -> int:
        res = self.__libtstat.tstat_export_core_statistics_read_chunk(self.__packets, 0)

        if res < 0:
            print("Tstat error #" + str(res))

        core_statistics_list_cursor = POINTER(TcsListElem)
        core_statistics_list_cursor = core_statistics_list_cursor.in_dll(self.__libtstat, "tcs_list_start")

        core_statistics_list_elements = 0
        core_statistics_list_elements_complete = 0
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
                row = core_statistics.get_row()
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
        # print("Number of tcp core statistics list elements: " + str(core_statistics_list_elements))
        # print(
        #     "Number of tcp core statistics list elements complete: " + str(core_statistics_list_elements_complete))
        # print("Number of tcp core statistics list elements nocomplete: " + str(
        #     core_statistics_list_elements_nocomplete))
        return res

    def analyze(self, file: str) -> Sequence[DataFrame]:
        """
        Analyzes a capture file.

        :param file: the file name to analyze
        :return: a list of dataframes where each dataframe contains the time steps of a flow
        """

        dictionary = {}

        pcap_file = c_char_p(file.encode('utf-8'))
        self.__libtstat.tstat_export_core_statistics_init(self.__conf_file, pcap_file, 0, 0, 0)

        res = 1
        while res == 1:
            res = self.read_tstat_chunk(dictionary)

        self.__libtstat.tstat_export_core_statistics_close(1)

        for flow_id in dictionary.keys():
            dictionary[flow_id] = DataFrame(columns=self.__dataframe_columns, data=dictionary[flow_id])
        return list(dictionary.values())

    def sniff(self, interface: str) -> Sequence[DataFrame]:
        """
        Sniffs an interface.

        :param interface: the name of the interface to sniff
        :return: a list of dataframes where each dataframe contains the time steps of a flow
        """

        dictionary = {}

        if self.__sniffing_initialized is False:
            interface_file = c_char_p(interface.encode('utf-8'))
            self.__libtstat.tstat_export_core_statistics_init(self.__conf_file,
                                                              interface_file,
                                                              1,  # live capture
                                                              int(self.__netgen_config.get("Tstat", "BufferSize")),
                                                              int(self.__netgen_config.get("Tstat", "Timeout")))
            self.__sniffing_initialized = True

        chunk_number = 1
        while chunk_number - 1 < int(self.__netgen_config.get("Tstat", "ChunksInDataframeList")):
            self.read_tstat_chunk(dictionary)
            chunk_number += 1

        dataframe_dictionary = {}
        for flow_id in dictionary.keys():
            if flow_id in self.__sniffing_last_dictionary.keys():
                last_dictionary_flow_stats = self.__sniffing_last_dictionary[flow_id]
                dictionary[flow_id][0:0] = last_dictionary_flow_stats
            dataframe_dictionary[flow_id] = DataFrame(columns=self.__dataframe_columns, data=dictionary[flow_id])
        self.__sniffing_last_dictionary = dictionary
        return list(dataframe_dictionary.values())

    def stop_sniff(self):
        """
        Stop sniffing an interface, closing gracefully the underlying Tstat instance.
        """
        self.__libtstat.tstat_export_core_statistics_close(1)
