import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

class Tokenizer():
    def __init__(self, texts, tokenizer, sos_token='<SOS>', eos_token='<EOS>', pad_token='<PAD>', unk_token='<UNK>'):
        
        self.tokenizer = word_tokenize if tokenizer == 'word' else sent_tokenize
        self.vocab, self.word2idx, self.idx2word = self.__build_vocab__(texts)
        self.sos_token, self.eos_token = self.word2idx[sos_token], self.word2idx[eos_token]

    def tokenize(self, text, sos_token='<SOS>', eos_token='<EOS>'):
        return [sos_token] + self.tokenizer(text) + [eos_token]
    
    def __build_vocab__(self, texts):
        vocab = set()
        for text in texts:
            for token in self.tokenize(text):
                vocab.add(token)
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}
        return vocab, word2idx, idx2word
    
    def padding(self, report, max_length):
        pad_report = self.tokenize(report)
        pad_report = [self.word2idx[word] if word in self.vocab else self.unk_token for word in pad_report]
        length = len(pad_report)
        if len(pad_report) > max_length:
            pad_report = pad_report[:max_length-1]
            pad_report += [self.eos_token]
        else:
            pad_report = list(pad_report) + [self.eos_token for _ in range(max_length - len(pad_report))]
        return pad_report, length
    
    def get_vocab_size(self):
        return len(self.word2idx)
    