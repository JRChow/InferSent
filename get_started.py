"""get_started.py: Sentence embedding using InferSent."""

# import nltk
# nltk.download('punkt')
import torch
from models import InferSent

__author__ = "ZHOU, JINGRAN"

RAW_MODEL_PATH = '/home/fyp1/language_style_transfer/code/is/InferSent' \
                 '/encoder/infersent%s.pkl'
SAMPLE_SENTENCES_PATH = '/home/fyp1/language_style_transfer/code/is' \
                        '/InferSent/encoder/samples.txt'
YELP_PATH = '/home/fyp1/language_style_transfer/data/dataset/ye/yelp.all'


class InferSentModel:
    def __init__(self, model_version, use_cuda, k):
        self.model = self.load_pretrained_model(model_version)
        if use_cuda:
            self.model = self.model.cuda()
        self.set_word_embedding(model_version)
        self.load_most_freq_embeddings(k)
        self.sentences = []
        self.embeddings = None

    @staticmethod
    def load_pretrained_model(model_version):
        model_path = RAW_MODEL_PATH % model_version
        params = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                  'pool_type': 'max', 'dpout_model': 0.0,
                  'version': model_version}
        model = InferSent(params)
        model.load_state_dict(torch.load(model_path))
        return model

    def set_word_embedding(self, model_version):
        w2v_path = 'dataset/GloVe/glove.840B.300d.txt' if model_version == 1 \
            else 'dataset/fastText/crawl-300d-2M.vec'
        self.model.set_w2v_path(w2v_path)

    # Load embeddings of K most frequent words
    def load_most_freq_embeddings(self, k):
        self.model.build_vocab_k_words(k)

    def load_sentences(self, file_path):
        with open(file_path) as f:
            for line in f:
                self.sentences.append(line.strip())
        print("Number of sentences: {0}".format(len(self.sentences)))

    def do_embedding(self):
        self.embeddings = self.model.encode(self.sentences, bsize=128,
                                            tokenize=False, verbose=True)
        print("Number of sentences encoded: {0}".format(len(self.embeddings)))


if __name__ == "__main__":
    version = 1  # The version trained with GloVe
    use_gpu = True  # To speed up
    most_freq_word_count = 100000
    model = InferSentModel(version, use_gpu, most_freq_word_count)
    model.load_sentences(SAMPLE_SENTENCES_PATH)
    model.do_embedding()
