"""
This script is for converting UCI format docword and vocab file to libsvm format data and dict

(How to run)

python text2libsvm.py <docword.input> <vocab.input> <libsvm.output> <dict.output>

"""

import sys

if len(sys.argv) != 5:
    print "Usage: python text2libsvm.py <docword.input> <vocab.input> <libsvm.output> <dict.output>"
    exit(1)

data_file = open(sys.argv[1], 'r')
vocab_file = open(sys.argv[2], 'r')

libsvm_file = open(sys.argv[3], 'w')
dict_file = open(sys.argv[4], 'w')

word_dict = {}
vocab_dict = []
doc = ""
last_doc_id = 0

line = vocab_file.readline()
while line:
    vocab_dict.append(line.strip())
    line = vocab_file.readline()

line = data_file.readline()
while line:
    col = line.strip().split(' ')
    if len(col) == 3:
        doc_id = int(col[0])
        word_id = int(col[1]) - 1
        word_count = int(col[2])
        if not word_dict.has_key(word_id):
            word_dict[word_id] = 0
        word_dict[word_id] += word_count
        if doc_id != last_doc_id:
            if doc != "":
                libsvm_file.write(doc.strip() + '\n')
            doc = str(doc_id) + '\t'
        doc += str(word_id) + ':' + str(word_count) + ' '
        last_doc_id = doc_id
    line = data_file.readline()

if doc != "":
    libsvm_file.write(doc.strip() + '\n')

libsvm_file.close()

for word in word_dict:
    line = '\t'.join([str(word), vocab_dict[word], str(word_dict[word])]) + '\n'
    dict_file.write(line)

