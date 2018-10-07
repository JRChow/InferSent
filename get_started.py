# - Some prerequisites
# import nltk
# nltk.download('punkt')
import torch

# - Load pre-trained model
from models import InferSent
V = 1  # The one trained with GloVe
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# - Keep it on CPU or GPU?
use_cuda = True
model = model.cuda() if use_cuda else model

# - Select word embedding (GloVe)
W2V_PATH = 'dataset/GloVe/glove.840B.300d.txt' if V == 1 else 'dataset/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)

# Load sentences (TODO)
# (First attempt) -> samples.txt
sentences = []
with open('encoder/samples.txt') as f:
	for line in f:
		sentences.append(line.strip())
print("Number of sentences: {0}".format(len(sentences)))

# Perform sentence embedding
embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
print("Number of sentences encoded: {0}".format(len(embeddings)))

