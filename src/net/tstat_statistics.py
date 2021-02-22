from ctypes import *

import socket
import struct


class TcpCoreStatistics(Structure):
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

    def __str__(self):
        printed_statistics = str(self.id_number) + " "
        printed_statistics += str(socket.inet_ntoa(struct.pack("<L", self.c2s_ip))) + " "
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
        printed_statistics += str(socket.inet_ntoa(struct.pack("<L", self.s2c_ip))) + " "
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
            printed_statistics += " NOCOMPLETE"
        else:
            printed_statistics += " COMPLETE"
        return printed_statistics

    def get_row(self):
        row = {'c2s_ip': self.c2s_ip, 'c2s_port': self.c2s_port, 'c2s_packets': self.c2s_packets,
               'c2s_reset_count': self.c2s_reset_count, 'c2s_ack_pkts': self.c2s_ack_pkts,
               'c2s_pureack_pkts': self.c2s_pureack_pkts, 'c2s_unique_bytes': self.c2s_unique_bytes,
               'c2s_data_pkts': self.c2s_data_pkts, 'c2s_data_bytes': self.c2s_data_bytes,
               'c2s_rexmit_pkts': self.c2s_rexmit_pkts, 'c2s_rexmit_bytes': self.c2s_rexmit_bytes,
               'c2s_out_order_pkts': self.c2s_out_order_pkts, 'c2s_syn_count': self.c2s_syn_count,
               'c2s_fin_count': self.c2s_fin_count, 's2c_ip': self.s2c_ip, 's2c_port': self.s2c_port,
               's2c_packets': self.s2c_packets, 's2c_reset_count': self.s2c_reset_count,
               's2c_ack_pkts': self.s2c_ack_pkts, 's2c_pureack_pkts': self.s2c_pureack_pkts,
               's2c_unique_bytes': self.s2c_unique_bytes, 's2c_data_pkts': self.s2c_data_pkts,
               's2c_data_bytes': self.s2c_data_bytes, 's2c_rexmit_pkts': self.s2c_rexmit_pkts,
               's2c_rexmit_bytes': self.s2c_rexmit_bytes, 's2c_out_order_pkts': self.s2c_out_order_pkts,
               's2c_syn_count': self.s2c_syn_count, 's2c_fin_count': self.s2c_fin_count, 'first_time': self.first_time,
               'last_time': self.last_time, 'completion_time': self.completion_time,
               'c2s_payload_start_time': self.c2s_payload_start_time, 'c2s_payload_end_time': self.c2s_payload_end_time,
               'c2s_ack_start_time': self.c2s_ack_start_time, 's2c_payload_start_time': self.s2c_payload_start_time,
               's2c_payload_end_time': self.s2c_payload_end_time, 's2c_ack_start_time': self.s2c_ack_start_time,
               'complete': self.complete, 'reset': self.reset, 'nocomplete': self.nocomplete}

        return row
