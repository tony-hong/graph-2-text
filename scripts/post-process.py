__author__ = "xhong@coli.uni-sb.com"


import os
import sys
import re
import gzip
import operator
import argparse
import json

from collections import OrderedDict as od
from collections import defaultdict
import xml.etree.cElementTree as et
import xml.dom.minidom as mdom

import numpy as np
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import replacer

detokenizer = TreebankWordDetokenizer()

# Configuration of environment
SRC_DIR = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
sys.path.append(SRC_DIR)


def lemmatize(word, pos=wn.VERB):
    lemma = wn.morphy(word, pos)
    return lemma if lemma else word

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def validate(token):
        '''validate whether the token is a valid word
        '''
        if re.search(r"[^a-zA-Z']+", token):
            return False
        return True

def penn2wn(pos):
    '''Converts P.O.S. tag from Penn TreeBank style to WordNet style
    '''
    first = pos[0]
    if first == 'J':
        return wn.ADJ
    elif first == 'N':
        return wn.NOUN
    elif first == 'R':
        return wn.ADV
    elif first == 'V':
        return wn.VERB
    return wn.NOUN



if __name__ == '__main__':
    prs = argparse.ArgumentParser(description="""
    This script converts conll format data from dependency parser and semantic role 
    labeller to xml format converted data.

    Parsed and SRL files must pair up correctly. 
    """)
    prs.add_argument('-l', '--lexicon',
                     help='Specify directory of all lexicons. ',
                     required=True)
    prs.add_argument('-b', '--labels',
                     help='Specify directory of all labels. ',
                     required=True)
    prs.add_argument('-s', '--sent',
                     help='Specify directory where target sentence file in CoNLL format are located. *** For test data, leave it empty ***', 
                     required=True)
    prs.add_argument('-o', '--out', default=os.getcwd(),
                     help='Specify output directory. If not specified, current '
                          'dir is used. ',
                     required=True)
    args = prs.parse_args()

    lexicon_dir = args.lexicon
    labels_dir = args.labels
    sent_dir = args.sent
    output_dir = args.out

    lexicon_fns = [os.path.join(lexicon_dir, name) for name in os.listdir(lexicon_dir) if re.search(r'json', name)]
    lexicon_fns.sort()
    print (len(lexicon_fns), lexicon_fns)

    labels_fns = [os.path.join(labels_dir, name) for name in os.listdir(labels_dir) if re.search(r'nodes', name)]
    labels_fns.sort()
    print (len(labels_fns), labels_fns)
    
    sent_fns = [os.path.join(sent_dir, name) for name in os.listdir(sent_dir) if re.search(r'txt', name)]
    sent_fns.sort()
    print (len(sent_fns), sent_fns)
    
    assert len(lexicon_fns) == len(sent_fns)
    
    for sent_fn_id, sent_fn in enumerate(sent_fns):
        print (sent_fn_id)
        lexicon_fn = lexicon_fns[sent_fn_id]
        labels_fn = labels_fns[sent_fn_id]
        print ('sent_fn:', sent_fn)
        print ('lexicon_fn:', lexicon_fn)
        print ('labels_fn:', labels_fn)
        
        output_name = re.findall(r'-lexicon/(.*)_DEEP\.entity\.json', lexicon_fn)[-1]
        # print ('output_name:', output_name)
        
        tokenised_sent_path = os.path.join(output_dir, 'T2-tokenized')
        untokenised_sent_path = os.path.join(output_dir, 'T2-detokenized')
        
        tokenised_sent_fn = os.path.join(tokenised_sent_path, output_name + '.txt')
        untokenised_sent_fn = os.path.join(untokenised_sent_path, output_name + '.txt')
        
        # token-based 
        print ('reading output sent ...')
        with open(sent_fn, 'r', encoding='utf-8') as f_sent:
            sent_data = f_sent.read().strip()
        sentences = sent_data.split('\n')
        
        print ('reading lexicon ...')
        with open(lexicon_fn, 'r', encoding='utf-8') as f_lex:
            lexion = json.load(f_lex)
        
        print ('reading output labels ...')
        with open(labels_fn, 'r', encoding='utf-8') as f_label:
            label_data = f_label.read().strip()
        labels = label_data.split('\n')
        
        print (len(labels))
        
        all_lexicon_keys = set()
        for lex_dict in lexion:
            lex_keys = lex_dict.keys()
            for key in lex_keys:
                all_lexicon_keys.add(key)
                
        print ('transforming input format...')
        tokenised_sent_list = []
        untokenised_sent_list = []
        
        for sent_id, sent in enumerate(sentences, start=0):
            tokens = sent.split(' ')            
            sent_entity_dict = lexion[sent_id]
            label_list = labels[sent_id]
            first_label = label_list.split('\t')[0]
            
            if len(tokens) == 0:
                tokens.append(first_label)
            
            no_dup_tokens = [tokens[0]]
            for idx in range(1, len(tokens)):
                token = tokens[idx]
                if tokens[idx] == tokens[idx-1]:
                    continue
                else:
                    no_dup_tokens.append(token)
            if len(no_dup_tokens) == 0:
                no_dup_tokens.append(first_label)
                
#             print ('sent_entity_dict', sent_entity_dict)
            lexicalised_tokens = []
            for token in no_dup_tokens:
                upper_token = token.upper()
                lexicalised_token = sent_entity_dict.get(upper_token, token)
                lexicalised_tokens.append(lexicalised_token)
                
            post_lexicalised_token = []
            for token in lexicalised_tokens:
                if token.upper() in all_lexicon_keys:
                    continue
                else:
                    post_lexicalised_token.append(token)                    
            if len(post_lexicalised_token) == 0:
                post_lexicalised_token.append(first_label)
                                
            lexicalised_sent = ' '.join(post_lexicalised_token)
            untokenised_sent = detokenizer.detokenize(post_lexicalised_token)

            if len(untokenised_sent) == 0:
                lexicalised_sent = first_label
                untokenised_sent = first_label
                
            # capitalise first char
            first_cap = untokenised_sent[0].upper()
            untokenised_sent = first_cap + untokenised_sent[1:]
            untokenised_sent += ' '

#             print ('before: lexicalised_sent', lexicalised_sent)
#             print ('before: untokenised_sent', untokenised_sent)
            
            lexicalised_sent = replacer.replacer(lexicalised_sent)
            untokenised_sent = replacer.replacer(untokenised_sent)

#             print ('lexicalised_sent', lexicalised_sent)
#             print ('untokenised_sent', untokenised_sent)
                
            tokenised_sent_list.append(lexicalised_sent)
            untokenised_sent_list.append(untokenised_sent)
            
        print('len(tokenised_sent_list)', len(tokenised_sent_list))
        print('len(untokenised_sent_list)', len(untokenised_sent_list))
        print('len(labels)', len(labels))
        
        assert len(tokenised_sent_list) == len(labels)
        assert len(untokenised_sent_list) == len(labels)
        assert len(untokenised_sent_list) == len(tokenised_sent_list)
        
        # do it in lower bcs. the eval is also in lower
        print ('writing tokenised sentences...')
        tokenised_sent_stream =''
        for idx, sent in enumerate(tokenised_sent_list, start=1):
            tokenised_sent_stream += '#sent_id = {0}\n'.format(idx)
            tokenised_sent_stream += '#text = {0}\n\n'.format(sent)
        
        '''
        tokenised_sent_stream = '\n'.join(tokenised_sent_list)
        tokenised_sent_stream += '\n'
        '''
        
        with open(tokenised_sent_fn, 'w', encoding='utf8') as f:
            f.write(tokenised_sent_stream)

        print ('writing untokenised sentences...')
        untokenised_sent_stream =''
        for idx, sent in enumerate(untokenised_sent_list, start=1):
            untokenised_sent_stream += '#sent_id = {0}\n'.format(idx)
            untokenised_sent_stream += '#text = {0}\n\n'.format(sent)
        
        '''
        untokenised_sent_stream = '\n'.join(untokenised_sent_list)
        untokenised_sent_stream += '\n'
        '''
        
        with open(untokenised_sent_fn, 'w', encoding='utf8') as f:
            f.write(untokenised_sent_stream)

            