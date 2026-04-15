from .base_tokenizer import BaseTokenizer
from .datatypes import Token
from .tiktoken_tokenizer import TiktokenTokenizer
from .hf_tokenizer import HFTokenizer

from typing import List
from dataclasses import dataclass, field
try:
    from transformers import AutoTokenizer
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

def get_tokenizer_impl(tokenizer_name: str = "cl100k_base") -> BaseTokenizer:
    return TiktokenTokenizer(model_name=tokenizer_name)

@dataclass
class Tokenizer(BaseTokenizer):
    """
    Encapsulates different tokenization implementations based on the specified model name.
    """

    model_name: str = "cl100k_base"
    _impl: BaseTokenizer = field(init=False, repr=False)

    def __post_init__(self):
        if not self.model_name:
            raise ValueError("TOKENIZER_MODEL must be specified in the ENV variables.")
        # TODO tokenizer impl的tokenizer_name传参可能有些问题
        # 具体来说, tokenizer_name的默认值是bpe编码的, 如果这里直接传qwen30b就会报错
        # 遇到bbpe怎么办呢? 后面考虑做一个映射规则
        self._impl = get_tokenizer_impl()

    def encode(self, text: str) -> List[int]:
        return self._impl.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self._impl.decode(token_ids)

    def count_tokens(self, text: str) -> int:
        return self._impl.count_tokens(text)