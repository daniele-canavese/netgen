"""
The MISP back-end.
"""
from datetime import datetime
from json import dumps
from json import loads

from pandas import DataFrame

from netgen.backends.backend import BackEnd
from netgen.misp import MISPEvent
from netgen.misp import MISPServer


class MISPBackEnd(BackEnd):
    """
    The MISP back-end.
    """

    def __init__(self, server: MISPServer, skip_fields: int) -> None:
        """
        Create the back-end.

        :param server: the MISP server
        :param skip_fields: the number of fields to skip in the results
        """

        self.__server = server
        self.__skip_fields = skip_fields

    def report(self, results: DataFrame, item: MISPEvent) -> None:
        """
        Reports some classification results.

        :param results: the classification results to report
        :param item: the object that triggered the classification
        """

        identifier = item.attributes["Attribute"]["event_id"]

        label = results.iloc[:, self.__skip_fields].mode()[0]
        confidence = results.iloc[:, self.__skip_fields + 1].mean()

        attacks = [{"attack_type": label, "confidence": confidence}]
        json = loads(item.attributes["Attribute"]["value"])
        json["NetGen"] = {
                "version":   "0.1",
                "reference": "https://github.com/daniele-canavese/netgen",
                "attacks":   attacks}
        item.attributes["Attribute"]["value"] = dumps(json)
        item.attributes["Attribute"]["timestamp"] = str(datetime.now().timestamp())
        self.__server.update_attributes(item.attributes)

        print(f"event {identifier} classified as {label} with confidence {confidence:.3}")
