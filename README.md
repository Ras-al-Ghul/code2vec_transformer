# Transformer Decoder model

Firstly, setup a **Python 3.6.0** virtualenv with the modules from `requirements.txt` installed.  
This repo consists of four directories:  
1. **Lexeminator** - contains scripts to clone SciTools/Understand (to get lexemes i.e. tokens from .java files) and setup the environment for it to run.  
Execute `./script.sh` and you're good to go. Possible change needed: In `users.txt`, change the username in the first column to the current user.  
2. **Transformer_Code_Attention** - contains code for the Transformer Attention model.  
   * Firstly execute `python lexeminator.py --src_folder=./DirWithAllThe.javaFiles` (uses the Understand module) to get two files - `lex_dumps_data.txt` (contains the training data encoded with numbers) and `vocab_file.txt` (contains a dict with the mapping between tokens and their number values) The data dump is after subtokenization of Identifiers.
   * `opt.py` (Adam optimizer), `model.py` (PyTorch model for the Transformer Decoder), `utils.py` (for data processing) are used by `train.py`
   * Once the data dump is ready, execute `mkdir save; mkdir ./save/code` (will save the embedding weight pickles in that directory) and then execute `python train.py --submit`. `--submit` is necessary to get weight pickles which are named as: `embed_weights_i.pt` where `i` is the epoch number. Validation loss - choosing 15% of the data randomly at each epoch - to check for overfitting has been incorporated.  
You could pass args like `--train_file=./lex_dumps_data.txt` (path to training file dump), `--n_embd=132` (embedding sizes), `--n_layer=9` (number of Transformer blocks), `--n_head=12` (number of heads for multihead attention), `--n_iter=60` (number of epochs), `--valid_percent=15` (15% of data for validation), `--n_ctx=22` (maximum number of tokens in a line), `--n_batch=500` (minibatch size)
   * To check GPU usage execute `watch -n 0.25 nvidia-smi`  
3. **Transformer_Postprocessing** - Contains code for *tSNE visualization* and dumping the learned embeddings as a dict for input to the *AST Paths* model. For tSNE plots, firstly `mkdir embeds_test`. Next, `python tSNE.py --vocab_file=VocabFileFromAbove --sess_dir=embeds_test --embed_file=PathTo_embed_weights_i.pt`.
Then `tensorboard --logdir=./embeds_test` and open up your browser.  
To get embeddings as a dict, `python to_dict.py` after setting the path to the *vocab_file.txt* and the *embed_weights_i.pt* in the *to_dict.py* file - it will output a `weights_dict.txt` file.  
4. **Transformer_tf_Sagemaker** - Contains code for the partially complete *Tensorflow Estimator* model which will be used by Sagemaker. To execute on local, `python transformer_sagemaker.py --train_file=PathTo_lex_dumps_data.txt`. Code for the custom Adam optimizer is added but does not work properly (loss increases at every step). One can also change `steps` in the .py file as there are no epochs in the estimator model. The issue right now is that the same weights are getting printed at every step - this could be due to two reasons - either the printing is wrong or the backprop part of the code in the estimator is not working as desired. The code has `#For Sagemaker` which means that those lines are required for hyperparameter tuning on Sagemaker.  
Refer to [OpenAI's tf code](https://github.com/openai/finetune-transformer-lm) and [huggingface's PyTorch port](https://github.com/huggingface/pytorch-openai-transformer-lm) for further details.