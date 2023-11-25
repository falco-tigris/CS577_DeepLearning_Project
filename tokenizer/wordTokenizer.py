import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

class WordTokenizer():
    def __init__(self, texts, tokenizer, vocab_size=400, sos_token='<SOS>', eos_token='<EOS>', pad_token='<PAD>', unk_token='<UNK>'):
        
        self.tokenizer = word_tokenize if tokenizer == 'word' else sent_tokenize
        self.vocab, self.word2idx, self.idx2word = self.__build_vocab__(texts, sos_token, eos_token, pad_token, unk_token)
        self.pad_token, self.unk_token = self.word2idx[pad_token], self.word2idx[unk_token]
        self.sos_token, self.eos_token = self.word2idx[sos_token], self.word2idx[eos_token] 
        self.vocab_size = len(self.vocab)

    def tokenize(self, text):
        return [self.idx2word[self.sos_token]] + self.tokenizer(text) + [self.idx2word[self.eos_token]]
    
    def __build_vocab__(self, texts, sos_token, eos_token, pad_token, unk_token):
        vocab = set()
        vocab.update([sos_token, eos_token, pad_token, unk_token])

        for text in texts:
            for token in self.tokenize(text):
                vocab.add(token)
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}
        return vocab, word2idx, idx2word
    
    def padding(self, report, max_length):
        pad_report = self.tokenize(report)
        if len(pad_report) > max_length:
            pad_report = pad_report[:max_length-1]
            pad_report += [self.eos_token]
        else:
            pad_report = list(pad_report) + [self.pad_token for _ in range(max_length - len(pad_report))]
        return pad_report
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_pad_token(self):
        return self.pad_token
    
    def get_unk_token(self):
        return self.unk_token
    