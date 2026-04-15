from dataclasses import dataclass
from typing import List

import tiktoken

from .base_tokenizer import BaseTokenizer


@dataclass
class TiktokenTokenizer(BaseTokenizer):
    def __post_init__(self):
        self.enc = tiktoken.get_encoding(self.model_name)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.enc.decode(token_ids)
