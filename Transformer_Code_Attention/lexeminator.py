# Understand (SciTools) code to extract tokens line by line
# and process them (remove whitespaces etc.)
import os
import sys
import argparse
import json
import shutil
import tempfile
import pprint
import understand as und
from re import finditer
from collections import OrderedDict

# create object and keep calling get_indexed_list() till it returns None
class tokenize_codebase():
    def __init__(self, src_folder, batch_size=20, language='java'):
        self.filelist = []
        self.filelistindex = 0
        self.token_list = []
        self.batch_size = batch_size   # number of sentences in a batch
        self.batch = []
        self.vocab_size = 0 # number of tokens in the vocabulary
        self.vocabulary_freq = {}    # counts frequencies of the tokens
        self.tempdir = tempfile.mkdtemp()
        self.language = language.lower()
        self.maxlen = 0  # maximum length of line
        self.linecount = 0   # counts number of lines until now
        self.overalllinecount = 0   # overall number of lines
        self.unk = 0    # unknown token
        self.src_folder = src_folder
        self.udb = self.build_udb()

        # other possible source code files
        self.extensions = {
            'java': '.java',
            'python': '.py',
            'csharp': '.cs',
            'c#': '.cs',
        }
        print("walking codebase")
        self.walk_codebase(self.src_folder)
        print("building vocab")
        self.build_vocab()
        print("init done")

        # sort the tokens according to frequencies in descending order
        self.valsorted_vocabulary_freq = sorted(self.vocabulary_freq.items(),
                                                key=lambda kv:kv[1], reverse=True)
        self.vocabulary_freq.clear()
        # assign indices to the tokens based on descending order of frequencies
        counter = 0
        for i in self.valsorted_vocabulary_freq:
            self.vocabulary_freq[i[0]] = counter
            if i[1] > 5:
                counter += 1
            else:
                del self.vocabulary_freq[i[0]]
                self.vocabulary_freq['unknown'] = counter
                break

        self.vocabulary_freq['start_tok'] = counter+1
        self.vocabulary_freq['end_tok'] = counter+2
        self.unk = counter
        
        data = json.dumps(OrderedDict(sorted(self.vocabulary_freq.items(),
                          key=lambda kv:kv[1])))
        with open('./vocab_file.txt', 'w') as f:
            f.write(data)
        f.close()
        # account for start and end tokens and because
        # counter starts at 0
        self.vocab_size = counter + 2 + 1
        # for start and end tokens
        self.maxlen += 2
        print('maxlen of sentence:', self.maxlen)
        print('len of training data:', self.overalllinecount)
        
    # create list of java files in the dir
    def walk_codebase(self, root):
        for root, dirs, files in os.walk(root):
            for file in files:
                if os.path.splitext(file)[-1] == \
                   self.extensions[self.language]:
                    self.filelist.append(os.path.join(root, file))

    # create the database and let Understand analyze the code
    def build_udb(self):
        list(map(lambda cmd: os.system(cmd), [
            f'und create -db {self.tempdir}/the.udb -languages {self.language} > /dev/null',
            f'und add -db {self.tempdir}/the.udb {self.src_folder} > /dev/null',
            f'und analyze -db {self.tempdir}/the.udb > /dev/null'
        ]))
        return und.open(os.path.join(self.tempdir, 'the.udb'))

    def camel(self, s):
        # https://stackoverflow.com/questions/10182664/check-for-camel-case-in-python
        return s != s.lower() and s != s.upper() and "_" not in s

    def snake_and_dash(self, s):
        return ("_" in s) or ("-" in s)

    def camel_case_split(self, identifier):
        # https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python
        matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]

    def snake_case_split(self, identifier):
        if "_" in identifier:
            return identifier.split('_')
        else:
            return identifier.split('-')

    # process the tokens
    def tokenize_file(self, filename):
        # do the cleaning here
        tokens = []
        for file in self.udb.ents('file'):
            if file.longname() == os.path.abspath(filename):
                # uncomment if you wish to see list of files as
                # they are being processed
                # print(file.longname(), len(self.vocabulary_freq))
                lexer = file.lexer()
                current_line = 0    # line number of file starts from 1

                tokens.append([])
                for lexeme in lexer:
                    if lexeme.line_begin() != current_line:
                        self.maxlen = max(self.maxlen, len(tokens[-1]))
                        if(tokens and tokens[-1] != []):
                            tokens[-1] = []
                            self.overalllinecount += 1
                        current_line = lexeme.line_begin()

                    if lexeme.token() != 'Comment' and \
                       lexeme.token() != 'Whitespace' and \
                       lexeme.token() != 'Newline':
                        tokens[-1].append([lexeme.text(), 
                                           lexeme.token()])

                        lexeme_list = []
                        if lexeme.token() == "Identifier":
                            if self.camel(lexeme.text()):
                                lexeme_list += (self.camel_case_split(lexeme.text()))
                            elif self.snake_and_dash(lexeme.text()):
                                lexeme_list += (self.snake_case_split(lexeme.text()))
                            else:
                                lexeme_list.append(lexeme.text())
                        else:
                            lexeme_list.append(lexeme.text())

                        for i in lexeme_list:
                            if not i in self.vocabulary_freq:
                                self.vocabulary_freq[i] = 1
                            else:
                                self.vocabulary_freq[i] += 1

                # after loop edge case
                if(tokens and tokens[-1] != []):
                    self.overalllinecount += 1

                break

    # wrapper function to traverse all files and build the vocabulary
    def build_vocab(self):
        print('number of files:', len(self.filelist))
        for file in self.filelist:
            self.tokenize_file(file)

    # index the tokens of given file
    def index_file(self, filename):
        # do the cleaning again
        tokens = []
        for file in self.udb.ents('file'):
            if file.longname() == os.path.abspath(filename):
                lexer = file.lexer()
                current_line = 0    # line number of file starts from 1

                tokens.append([self.vocabulary_freq['start_tok']])
                for lexeme in lexer:
                    if lexeme.line_begin() != current_line:
                        if(tokens and (tokens[-1] != 
                                       [self.vocabulary_freq['start_tok']])):
                            tokens[-1].append(self.vocabulary_freq['end_tok'])
                            self.token_list.append(tokens[-1])
                            tokens.append([])
                            tokens[-1].append(self.vocabulary_freq['start_tok'])
                        current_line = lexeme.line_begin()

                    if lexeme.token() != 'Comment' and \
                       lexeme.token() != 'Whitespace' and \
                       lexeme.token() != 'Newline':
                        # replace with integer values from the dict
                        lexeme_list = []
                        if lexeme.token() == "Identifier":
                            if self.camel(lexeme.text()):
                                lexeme_list += (self.camel_case_split(lexeme.text()))
                            elif self.snake_and_dash(lexeme.text()):
                                lexeme_list += (self.snake_case_split(lexeme.text()))
                            else:
                                lexeme_list.append(lexeme.text())
                        else:
                            lexeme_list.append(lexeme.text())

                        for i in lexeme_list:
                            if i in self.vocabulary_freq:
                                tokens[-1].append(self.vocabulary_freq[i])
                            else:
                                tokens[-1].append(self.unk)
                # after loop edge case                        
                if(tokens and (tokens[-1] != [self.vocabulary_freq['start_tok']])):
                    tokens[-1].append(self.vocabulary_freq['end_tok'])
                    self.token_list.append(tokens[-1])

    # return indexed list of tokens
    # from multiple files, with total number of sentences < maxline
    def get_indexed_list(self):
        # don't count previous values
        from_prev_call = len(self.token_list)
        while self.filelistindex < len(self.filelist):
            self.index_file(self.filelist[self.filelistindex])
            self.filelistindex += 1
            if len(self.token_list) >= self.batch_size or \
               self.filelistindex == len(self.filelist):
                # update global linecount
                self.linecount += (len(self.token_list) - 
                                   from_prev_call)
                return self.token_list
        return None

    # gets x_lines which will be a part of one epoch
    # where x is self.batch_size
    def get_x_lines(self):
        # fill self.token_list if its length is less than batch_size
        if(len(self.token_list) < self.batch_size):
            ret_val = self.get_indexed_list()
            if ret_val == None and len(self.token_list)==0:
                return False
        # edge case: last incomplete list of sentences
        if len(self.token_list) < self.batch_size:
            ret_val = self.token_list[:]
            del self.token_list[:]
            return ret_val
        # normal case
        ret_val = self.token_list[:self.batch_size]
        del self.token_list[:self.batch_size]
        return ret_val

    # gets all the lines at once to create datadump
    def get_all_lines(self):
        trX = True
        ret = []
        while trX:
            trX = self.get_x_lines()
            ret.append(trX)
        return ret

    # close temp dirs
    def clean_up(self):
        self.udb.close()
        shutil.rmtree(self.tempdir)

    # returns vocabsize
    def get_vocabsize(self):
        return self.vocab_size

    # returns total number of lines counted until now
    def get_linecount(self):
        return self.linecount

    # returns overall number of lines in the training set
    def get_overalllinecount(self):
        return self.overalllinecount

    # returns max number of tokens in a sentence
    def get_maxlen(self):
        return self.maxlen

    # gets index for particular term in the vocabulary
    def get_val(self, key):
        if key in self.vocabulary_freq:
            return self.vocabulary_freq[key]
        else:
            return None

    # returns the vocabulary dict itself
    def get_vocab_dict(self):
        return self.vocabulary_freq

    # returns unknown token
    def get_unk(self):
        return self.unk
    
    # for next epoch
    def reset_filelist_index(self):
        self.filelistindex = 0


def main():
    '''
    src_folder is the folder with all the .java files
    
    n_max_number_of_lines is the very generous estimate of
    the max_number_of_lines of code in all the files of the
    src_folder - let it be a very high number
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_folder', type=str, default='../tr_dir/')
    parser.add_argument('--n_max_number_of_lines', type=int, default=40000000)
    args = parser.parse_args()
    globals().update(args.__dict__)
    code_encoder = tokenize_codebase(src_folder=src_folder, 
                                     batch_size=n_max_number_of_lines)
    with open('./lex_dumps_data.txt', 'w') as f:
        n_vocab = code_encoder.get_vocabsize()
        trX = code_encoder.get_all_lines()
        trX = trX[0]    # remove the none
        trX = [[n_vocab]] + trX   # add the vocab size as first line
        # to help with deciding n_ctx size
        count_20 = 0
        count_25 = 0
        count_30 = 0
        maxs = 0
        for i in range(len(trX)):
            maxs = max(maxs, len(trX[i]))
            if(len(trX[i]) > 20):
                count_20 += 1
            if(len(trX[i]) > 25):
                count_25 += 1
            if(len(trX[i]) > 30):
                count_30 += 1
            f.write(str(trX[i])+'\n')
    print('lines with greater than 20 tokens:', count_20)
    print('lines with greater than 25 tokens:', count_25)
    print('lines with greater than 30 tokens:', count_30)

if __name__ == '__main__':
    main()