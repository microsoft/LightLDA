#!/bin/bash

root=`pwd`
echo $root
bin=$root/../bin
dir=$root/data/pubmed

mkdir -p $dir
cd $dir

# 1. Download the data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.pubmed.txt.gz
gunzip $dir/docword.pubmed.txt.gz
wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.pubmed.txt

# 2. UCI format to libsvm format
python $root/text2libsvm.py $dir/docword.pubmed.txt $dir/vocab.pubmed.txt $dir/pubmed.libsvm $dir/pubmed.word_id.dict

# 3. libsvm format to binary format
$bin/dump_binary $dir/pubmed.libsvm $dir/pubmed.word_id.dict $dir 0

# 4. Run LightLDA
$bin/lightlda -num_vocabs 144400 -num_topics 1000 -num_iterations 100 -alpha 0.1 -beta 0.01 -mh_steps 4 -num_local_workers 1 -num_blocks 1 -max_num_document 8300000 -input_dir $dir -data_capacity 6200
