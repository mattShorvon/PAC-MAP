import time
from typing import List, Tuple, Optional
from spn.utils.evidence import Evidence
import math


class AnytimeInfo:
    def __init__(self):
        self.__starting_time = time.process_time()
        self.__lowers: List[Tuple[Evidence, float, float]] = []  # evidence, value, time
        self.__finished = False

    def new_lower_bound(self, value: float, evidence: Evidence):
        try:
            self.__lowers.append(
                (evidence, math.log(value), time.process_time() - self.__starting_time)
            )
        except ValueError:  # Log of 0 causes value error
            if len(self.__lowers) > 0:
                self.__lowers.append(
                    (
                        evidence,
                        min(x[1] for x in self.__lowers) - 10,
                        time.process_time() - self.__starting_time,
                    )
                )
            else:
                self.__lowers.append(
                    (evidence, -10000, time.process_time() - self.__starting_time)
                )

    def best_evidence(self) -> Evidence:
        return self.__lowers[-1][0]

    def finish(self):
        self.__finished = True

    @property
    def lowers(self) -> List[Tuple[Evidence, float, float]]:
        return self.__lowers

    @property
    def finished(self) -> bool:
        return self.__finished
