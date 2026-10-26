import string

def word_frequency(filepath: str) -> dict:
    # Create a translation table that maps punctuation to None
    translator = str.maketrans('', '', string.punctuation)
    frequency = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Remove punctuation and convert to lower case
            line = line.translate(translator).lower()
            words = line.split()
            for word in words:
                frequency[word] = frequency.get(word, 0) + 1
    return frequency
