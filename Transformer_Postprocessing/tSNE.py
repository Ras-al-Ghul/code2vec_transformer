import torch
import json
import argparse
import numpy as np
import tensorflow as tf

p = tf.contrib.tensorboard.plugins.projector

def main():
	# takes as input the vocab_file of the code tokens
	# and the learned embedding_weights.pt file and creates a 
	# session directory for tensorboard
	global p
	parser = argparse.ArgumentParser()
	parser.add_argument('--vocab_file', type=str, 
						default='../Transformer_Code_Attention/vocab_file.txt')
	parser.add_argument('--sess_dir', type=str, default='./embeds_test')
	parser.add_argument('--embed_file', type=str, default='./embed_weights_3.pt')
	args = parser.parse_args()
	globals().update(args.__dict__)

	l = None
	with open(vocab_file, 'r') as f:
		l = f.read()
		l = json.loads(l)
	vocab_size = len(l)
	print('vocab_size:', vocab_size)
	with open('./'+sess_dir+'/vocabs.csv','w') as f:
		for k,v in l.items():
			f.write(k+'\n')
	model = torch.load(embed_file, map_location='cpu')
	print(model)
	# convert pytorch tensor to numpy array
	np_weights = model.detach().numpy()
	print('old_shape:', np_weights.shape)
	# needed if there is a mismatch in the vocab size and embedding rows
	# will be known if the tensorboard throws an error
	# np_weights = np.delete(np_weights, 3362, axis=0)
	# print(np_weights.shape)
	np_weights = np.delete(np_weights, [i for i in range(vocab_size, vocab_size+200)], axis=0)
	print('new_shape:', np_weights.shape)
	# convert numpy array to tf tensor
	tf_weights = tf.convert_to_tensor(np_weights)
	print(tf_weights)
	
	# https://www.lewuathe.com/t-sne-visualization-by-tensorflow.html
	# refer for details
	embedding_var = tf.Variable(tf_weights, name='embedding')
	with tf.Session() as sess:
		writer = tf.summary.FileWriter(sess_dir, sess.graph)
		sess.run(embedding_var.initializer)
		config = p.ProjectorConfig()
		embedding = config.embeddings.add()
		embedding.tensor_name = embedding_var.name
		embedding.metadata_path = 'vocabs.csv'
		p.visualize_embeddings(writer, config)
		saver_embed = tf.train.Saver([embedding_var])
		saver_embed.save(sess, './'+sess_dir+'/embedding.ckpt', 1)
		writer.close()

if __name__ == '__main__':
	main()

