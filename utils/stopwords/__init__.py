__all__ = [
    'get_medical_stopwords',
    'get_custom_stopwords',
    'get_english_stopwords'
]

from utils.stopwords.medical import get_medical_stopwords
from utils.stopwords.generic import get_custom_stopwords, get_english_stopwords
