import math
from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class Token:
    text: str
    prob: float
    top_candidates: List = field(default_factory=list)
    ppl: Union[float, None] = field(default=None)

    @property
    def logprob(self) -> float:
        return math.log(self.prob)