def tokens(self, tokenization):
    self.tokenization = tokenization

    if isinstance(tokenization, str):
        texts = [tokenization]
    elif isinstance(tokenization, list):
        texts = tokenization
    else:
        raise TypeError("tokenization must be a string or a list of strings")

    # split by whitespace
    tokenized_texts = [text.split() for text in texts]

    # flatten if only one input string, else keep list of lists
    self.tokenized = tokenized_texts[0] if isinstance(tokenization, str) else tokenized_texts

    return self.tokenized
