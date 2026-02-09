"""
Simple tokenizer for Tiny BERT
Converts words to numeric IDs for model input
"""


class SimpleTokenizer:
    """Convert words to numbers"""

    def __init__(self):
        self.word_to_id = {"[PAD]": 0, "[UNK]": 1}
        self.id_to_word = {0: "[PAD]", 1: "[UNK]"}
        self.next_id = 2

    def add_word(self, word):
        """Add a new word to vocabulary"""
        if word not in self.word_to_id:
            self.word_to_id[word] = self.next_id
            self.id_to_word[self.next_id] = word
            self.next_id += 1

    def encode(self, text, max_length=10, add_new_words=True):
        """Convert text to word IDs"""
        words = text.lower().split()

        # Add words to vocabulary only during training
        if add_new_words:
            for word in words:
                self.add_word(word)

        # Convert to IDs (unknown words map to [UNK] = 1)
        ids = [self.word_to_id.get(word, 1) for word in words]

        # Pad or truncate
        if len(ids) < max_length:
            ids += [0] * (max_length - len(ids))
        else:
            ids = ids[:max_length]

        return ids

    def decode(self, ids):
        """Convert IDs back to text"""
        words = [self.id_to_word.get(id, "[UNK]") for id in ids if id != 0]
        return " ".join(words)
