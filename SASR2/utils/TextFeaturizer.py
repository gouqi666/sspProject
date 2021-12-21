from .config import lm_token_path


class TextFeaturizer:
    def __init__(self, token_path):
        # token格式：每行一个单词，最开始两个是开始和结束符号，最后一行为#，读取到此停止
        self.word2token = {}
        self.token2word = {}
        self.init_dict(token_path)

    def init_dict(self, token_path):
        with open(token_path, 'r', encoding='utf8') as fp:
            lines = fp.readlines()
            self.word2token["[PAD]"] = 0
            self.token2word[0] = ""
            for idx, line in enumerate(lines):
                line = line.strip()
                if line != "#":
                    self.word2token[line] = idx + 1
                    self.token2word[idx + 1] = line
                else:
                    break
            self.word2token["[UNS]"] = len(lines)
            self.token2word[len(lines)] = "[UNS]"

    @property
    def unknown_token(self):
        return self.word2token["[UNS]"]

    @property
    def blank_token(self):
        return self.word2token["[PAD]"]

    @property
    def start_token(self):
        return self.word2token["<S>"]

    @property
    def end_token(self):
        return self.word2token["</S>"]

    @property
    def vocabulary(self):
        return list(self.word2token.keys())

    @property
    def vocab_size(self):
        return len(self.token2word)

    def encode(self, words):
        res = [self.start_token]
        for word in words:
            if word in self.word2token:
                res.append(self.word2token[word])
            else:
                res.append(self.unknown_token)
        res.append(self.end_token)
        return res

    def decode(self, tokens):
        return [self.token2word[token] for token in tokens][1:-1]


if __name__ == '__main__':
    print(TextFeaturizer(lm_token_path).encode("我是大帅哥"))
