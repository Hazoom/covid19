import re

CURRENCIES = {'$': 'USD', 'zł': 'PLN', '£': 'GBP', '¥': 'JPY', '฿': 'THB',
              '₡': 'CRC', '₦': 'NGN', '₩': 'KRW', '₪': 'ILS', '₫': 'VND',
              '€': 'EUR', '₱': 'PHP', '₲': 'PYG', '₴': 'UAH', '₹': 'INR'}

RE_NUMBER = re.compile(
    r"(?:^|(?<=[^\w,.]))[+–-]?"
    r"(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)"
    r"(?:$|(?=\b))")

RE_URL = re.compile(
    r'((http://www\.|https://www\.|http://|https://)?' +
    r'[a-z0-9]+([\-.][a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(/.*)?)')

# English Stop Word List (Standard stop words used by Apache Lucene)
STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
              "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
              "they", "this", "to", "was", "will", "with"}
