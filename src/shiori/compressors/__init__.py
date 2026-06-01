from shiori.compressors.base import CompressionResult, Compressor
from shiori.compressors.caveman import CavemanCompressor
from shiori.compressors.dictionary import DictionaryCompressor
from shiori.compressors.template import TemplateCompressor
from shiori.compressors.lossless import LosslessCompressor

__all__ = [
    "CompressionResult",
    "Compressor",
    "CavemanCompressor",
    "DictionaryCompressor",
    "TemplateCompressor",
    "LosslessCompressor",
]
