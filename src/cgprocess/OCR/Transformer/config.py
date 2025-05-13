"""
Configuration file for Transformer based OCR.
"""

# Tokenizer/Dataset parameter
ALPHABET = ['<PAD>', '<START>', '<NAN>', '<END>',   # do not change these 4 tokens!
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ſ', 'ß', 'à', 'á', 'è', 'é', 'ò', 'ó', 'ù', 'ú',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            ' ', ',', '.', '?', '!', '-', '—', '_', ':', ';', '/', '\\', '(', ')', '[', ']', '{', '}',
            '%', '$', '£', '§', '\"', '„', '“', '»', '«', '\'', '’', '&', '+', '~', '*', '=', '†']

MAX_SEQUENCE_LENGTH = 512
PAD_HEIGHT = 64
PAD_WIDTH = 16


# Training parameter
LR = 1e-4
LOG_EVERY = 1280
BATCH_SIZE = 128
