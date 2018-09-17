import torch
import numpy as np
import tensorflow as tf
import json

np.set_printoptions(threshold=np.nan)

def main():
	# input is a vocabfile contaning the tokens in the vocabulary
	# and the embedding_weights.pt file which contains the learned
	# embeddings and it outputs weights along with theiry code token
	# keys in the form of a dictionary so that it can be easily loaded
	# into the AST paths model
	l = None
	with open('../Transformer_Code_Attention/vocab_file.txt', 'r') as f:
		l = f.read()
		l = json.loads(l)
	vocab_size = len(l)
	model = torch.load('./embed_weights_3.pt', map_location='cpu')
	print('old_shape:', model.size)
	np_weights = model.detach().numpy()
	np_weights = np.delete(np_weights, [i for i in range(vocab_size,vocab_size+200)], axis=0)
	print('new_shape:', np_weights.shape)
	dict_of_shapes = {}
	for k, v in l.items():
		dict_of_shapes[k] = np_weights[v][:].tolist()
	data = json.dumps(dict_of_shapes)
	with open('weights_dict.txt', 'w') as f:
		f.write(data)

if __name__ == '__main__':
	main()

