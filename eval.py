#!/usr/bin/env python

"""
	Run eval for every eval with configs specified in JSON file.

"""

import os
import subprocess
import sys
import logging
import json
import glob


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
	Imports from eval script
"""
import codecs
import collections
import io
import os
import sys
import glob
try:
    from nltk.metrics import *
    import nltk.translate.nist_score as ns
    import nltk.translate.bleu_score as bs
except ImportError:
    print('Please install nltk (https://www.nltk.org/)')
    print("For instance: 'sudo pip install -U nltk\'")
    exit()


def read_corpus(filename, ref=False, normalize=True):
    """Reads a corpus

    Args:
      filename: Path and file name for the corpus.

    Returns:
      A list of the sentences.
    """
    data = []
    #print('Received filename: ', filename)
    with open(filename, 'r', encoding='utf-8') as f:
        #for line in codecs.getreader('utf-8')(f, errors='ignore'):  # type: ignore
        for line in codecs.open(f.name, 'r', 'utf-8'):
            line = line.rstrip()
            if line.startswith(u'# text'):
                split = line.split(u'text = ')
                if len(split) > 1:
                    text = split[1]
                else:
                    text = '# #'
                if normalize:
                    text = text.lower()
                if ref:
                    data.append([text.split()])
                else:
                    data.append(text.split())
    return data
    print(split)

def read_corpus_lines(filename, ref=False, normalize=True):
    data = []
    for line in open(filename,'r', encoding='utf-8').readlines():
        line = line.replace('\n','').lower()
        if ref:
            data.append([line.split()])
        else:
            data.append(line.split())
    return data


def run_eval(system_path,ref_path):
    # For all files in system path.
    logger.info('Evaluating output file {0}'.format(system_path))
    system_filename = system_path
    ref_filename = ref_path

    ref = read_corpus_lines(ref_filename, ref=True)
    hyp = read_corpus_lines(system_filename, ref=False)

    # NIST score
    nist = ns.corpus_nist(ref, hyp, n=4)

    # BLEU score
    chencherry = bs.SmoothingFunction()
    bleu = bs.corpus_bleu(ref, hyp, smoothing_function=chencherry.method2)
    logger.info('BLEU', str(round(bleu, 3)))
    total_len = 0.0
    edi = 0.0
    for r, h in zip(ref, hyp):
        total_len += max(len(r[0]), len(h))
        edi += edit_distance(r[0], h)
    logger.info('DIST', str(round(1-edi/total_len,3)))
    logger.info('NIST', str(round(nist, 6)))


def exe(cmd):
    subprocess.call(cmd,stdout=subprocess.PIPE,shell=True)

def translate(model_path,output_path,gpu):
    translate_cmd = 'python3 translate.py \
                -src processed_corpus/SR19_T2_delex_test/test_T2-test_en_ewt-ud-test_DEEP.delex-src-nodes.txt \
                -tgt processed_corpus/SR19_T2_delex_test/test_T2-test_en_ewt-ud-test_DEEP.output.dat \
                -src_label processed_corpus/SR19_T2_delex_test/test_T2-test_en_ewt-ud-test_DEEP.delex-src-labels.txt \
                -src_node1 processed_corpus/SR19_T2_delex_test/test_T2-test_en_ewt-ud-test_DEEP.delex-src-node1.txt \
                -src_node2 processed_corpus/SR19_T2_delex_test/test_T2-test_en_ewt-ud-test_DEEP.delex-src-node2.txt \
                -replace_unk \
                -verbose -dynamic_dict -batch_size 1 \
                -max_length 5 -block_ngram_repeat 3 \
                -model {0} \
                -gpu {2} \
                -output {1} \
                -data_type gcn'.format(model_path,output_path,gpu) 
    exe(translate_cmd)

def train(configs):
    train_cmd = 'python3 train.py \
                -data {0} \
                -save_model {1} \
                -rnn_size 512 \
                -word_vec_size 512 \
                -layers {2} \
                -epochs {3} \
                -optim adam \
                -learning_rate 0.001 \
                -dropout 0.5 \
                -encoder_type gcn \
                -gcn_num_inputs 512 \
                -gcn_num_units 512 \
                -gcn_in_arcs \
                -gcn_out_arcs \
                -gcn_num_layers 4 \
                -gcn_num_labels 19 \
                -gcn_residual residual \
                -copy_attn \
                -reuse_copy_attn \
                -gpuid {4}'.format(configs['data'],configs['save_model'],configs['layers'],configs['epoch'],configs['gpuid'])
    exe(train_cmd)
 
def score(output_path,ref_dir):
    make_temp = 'cp {0} tmp/temp'.format(output_path)
    rm_temp = 'rm tmp/temp'.format(output_path)
    exe(make_temp)
    run_eval(
        system_path='tmp/temp',
        ref_path=ref_dir
    )
    exe(rm_temp) 


def evaluate(configs):
    
    '''
    model_path = 'save/SR19_T2_GCN_4L_reuse_AND_copy_attn/SR19_T2_GCN_4L_reuse_AND_copy_attn_acc_50.26_ppl_18.29_e12.pt'
    sys_path = 'system'
    ref_path = 'gold'
    ''' 

    logger.info('\tInitializing evaluation...')
    translate(
        model_path=configs['model_path'],
        output_path=configs['sys_path'],
        gpu=configs['gpu']
    )
    logger.info('\tUsing GPU {0}'.format(configs['gpu']))
    logger.info('\tComputing score...') 
    score(
        output_path=configs['sys_path'],
        ref_dir=configs['ref_path'] 
    )
    

if __name__ == '__main__':
    OPTION = sys.argv[1]
    CONFIGS_FILE = sys.argv[2]
    configs = json.load(open(CONFIGS_FILE, 'r'))
    if OPTION == 'train':
        train(configs)
    else:
        evaluate(configs)
    





  
