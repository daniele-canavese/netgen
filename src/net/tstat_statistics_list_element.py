from ctypes import *

from .tstat_statistics import TcpCoreStatistics


class TcsListElem(Structure):
    pass


TcsListElem._fields_ = [("next", POINTER(TcsListElem)),
                        ("prev", POINTER(TcsListElem)),
                        ("stat", POINTER(TcpCoreStatistics))]
