#! /usr/bin/python

''' this script is used to count the tf information from
    libsvm data
    Usage:
        python get_meta.py <libsvm_input_data> <output_file>
'''

import sys
import re

finput = open(sys.argv[1], 'r')

word_dict = {}

line = finput.readline()
while line:
    doc = re.split(" |\t", line.strip())[1:]
    for word_count in doc:
        col = word_count.strip().split(":")
        if len(col) != 2:
            print "error!"
        if not word_dict.has_key(int(col[0])):
            word_dict[int(col[0])] = 0
        word_dict[int(col[0])] += int(col[1])
    line = finput.readline()

foutput = open(sys.argv[2], 'w')
for word in word_dict:
    line = '\t'.join([str(word), "word", str(word_dict[word])]) + '\n'
    foutput.write(line)


