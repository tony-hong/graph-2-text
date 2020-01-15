# conll2xml.py
__author__ = "xhong@coli.uni-sb.com"


import os
import sys
import re
import gzip
import operator
import argparse
import pickle

from collections import OrderedDict as od
from collections import defaultdict
import xml.etree.cElementTree as et
import xml.dom.minidom as mdom

import numpy as np
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize


# Configuration of environment
SRC_DIR = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
sys.path.append(SRC_DIR)


if __name__ == '__main__':
    prs = argparse.ArgumentParser(description="""
    This script converts conll format data from dependency parser and semantic role 
    labeller to xml format converted data.

    Parsed and SRL files must pair up correctly. 
    """)
    # prs.add_argument('-d', '--dir',
    #                  help='Specify directory where conll files are located.',
    #                  required=False)
    prs.add_argument('-g', '--graph',
                     help='Specify directory where graph file in CoNLL format are located. ',
                     required=True)
    prs.add_argument('-s', '--sent',
                     help='Specify directory where target sentence file in CoNLL format are located. *** For test data, leave it empty ***', 
                     required=False)
    prs.add_argument('-o', '--out', default=os.getcwd(),
                     help='Specify output directory. If not specified, current '
                          'dir is used.',
                     required=True)
    prs.add_argument('-p', '--propertyType', default=False,
                       help='Add properties to nodes. ')
    prs.add_argument('-b', '--seq2seqAblation', action='store_true',
                       help='Ablation study for seq2seq. ')
    args = prs.parse_args()
    
    graph_dir = args.graph
    if not args.sent:
        sent_dir = graph_dir
        noOutput = True
    else:
        sent_dir = args.sent
        noOutput = False

    if args.seq2seqAblation:
        from utils_ablation import CoNLLTools, PROPERTY_MAP
    else:
        from utils import CoNLLTools, PROPERTY_MAP
        # from utils_merge_inv import CoNLLTools, PROPERTY_MAP

    if args.propertyType:
        property_vocab = PROPERTY_MAP[args.propertyType]
        
    graph_fns = [os.path.join(graph_dir, name) for name in os.listdir(graph_dir) if re.search(r'conllu', name)]
    graph_fns.sort()
    print (graph_fns)
    
    # parse_fns = [os.path.join(args.parse, name) for name in os.listdir(args.parse) if re.search(r'conll', name)]
    # parse_fns.sort()
    # print (parse_fns)

    sent_fns = [os.path.join(sent_dir, name) for name in os.listdir(sent_dir) if re.search(r'conllu', name)]
    sent_fns.sort()
    print (sent_fns)
    
    ct = CoNLLTools()
    
    all_graph_data = ''
    all_sent_data = ''
    for graph_id, graph_fn in enumerate(graph_fns):
        print (graph_fn)
        sent_fn = sent_fns[graph_id]

        output_name = re.findall(r'SR19/.*/(.*)\.conllu', graph_fn)[-1]
        output_name = output_name.replace('/', '_')
        # a hack to change the file name
        output_name = output_name.replace('partut', 'all')
        output_name = output_name.replace('gsd', 'all')
        output_name = output_name.replace('sequoia', 'all')
        
        dat_input_name = output_name + '.input.dat'
        dat_output_name = output_name + '.output.dat'
        entity_output_name = output_name + '.entity.json'
        input_file_fn = os.path.join(args.out, dat_input_name)        
        output_file_fn = os.path.join(args.out, dat_output_name)
        entity_file_fn = os.path.join(args.out, entity_output_name)
        
        nodes_file_name = output_name + '.delex-src-nodes.txt'
        feats_file_name = output_name + '.delex-src-feats.txt'
        labels_file_name = output_name + '.delex-src-labels.txt'
        node1s_file_name = output_name + '.delex-src-node1.txt'
        node2s_file_name = output_name + '.delex-src-node2.txt'
        nodes_file_fn = os.path.join(args.out, nodes_file_name)
        feats_file_fn = os.path.join(args.out, feats_file_name)
        labels_file_fn = os.path.join(args.out, labels_file_name)
        node1s_file_fn = os.path.join(args.out, node1s_file_name)
        node2s_file_fn = os.path.join(args.out, node2s_file_name)

        # print (output_name)
        # print (output_path)

        print ('reading input semantic graph ...')
        graph_data = ct.gzip_reader(graph_fn)
        
        # token-based 
        print ('reading output sent ...')
        sent_data = ct.gzip_reader(sent_fn)
        
        all_graph_data += graph_data
        all_sent_data += sent_data
        
    print ('extracting input dataframe ...')
    graph_dfs, _ = ct.extract_dataframe(all_graph_data, toLower=False)
    
    print ('extracting output dataframe ...')
    sent_dfs, sentences = ct.extract_dataframe(all_sent_data, toLower=False)
    
    '''
    parse_data = ct.gzip_reader(parse_fn)
    print ('extracting dataframe ...')
    parse_dfs = ct.extract_dataframe(parse_data, toLower=False)
    '''
    
    print ('transforming input format...')
    result_input_list = []
    result_output_list = []
    entity_dict_list = []

    nodes_list = []
    feats_list = []
    labels_list = []
    node1s_list = []
    node2s_list = []

    all_property_keys = set()
    for idx_df, graph_df in enumerate(graph_dfs, start=0):
        if noOutput:
            sent = '<NO OUTPUT>'
        else:
            sent = sentences[idx_df]

        # parse_df = parse_dfs[idx_df]

        # print (graph_df[0])
        # print (sent)

        input_data, output_sent, entity_lemma_dict, nodes_line, feats_line, labels_line, node1_line, node2_line, property_keys = ct.df_2_dat_DEEP(graph_df, sent, noOutput, property_vocab)

        if not input_data:
            continue

        result_input_list.append(input_data)
        result_output_list.append(output_sent)
        entity_dict_list.append(entity_lemma_dict)

        nodes_list.append(nodes_line)
        feats_list.append(feats_line)
        labels_list.append(labels_line)
        node1s_list.append(node1_line)
        node2s_list.append(node2_line)

        all_property_keys = all_property_keys.union(property_keys)
        
        # DEBUG block
        # print (input_df)
        # print (new_data)
        # if idx_df > 50:
            # break

    print ('All properties: ', all_property_keys)
    
    print ('writing input data...')
    input_file_stream = '\n'.join(result_input_list)
    input_file_stream += '\n'
    with open(input_file_fn, 'w',  encoding='utf8') as f:
        f.write(input_file_stream)

    # do it in lower bcs. the eval is also in lower
    print ('writing output data...')
    output_file_stream = '\n'.join(result_output_list)
    output_file_stream += '\n'
    with open(output_file_fn, 'w', encoding='utf8') as f:
        f.write(output_file_stream.lower())

    print ('writing entity list...')
    import json
    with open(entity_file_fn, 'w') as f:
        json.dump(entity_dict_list, f)            
    
    
    # write GCN files
    print ('writing output nodes...')
    nodes_file_stream = '\n'.join(nodes_list)
    nodes_file_stream += '\n'
    with open(nodes_file_fn, 'w', encoding='utf8') as f:
        f.write(nodes_file_stream)
        
    print ('writing output feats...')
    feats_file_stream = '\n'.join(feats_list)
    feats_file_stream += '\n'
    with open(feats_file_fn, 'w', encoding='utf8') as f:
        f.write(feats_file_stream)
        
    print ('writing output labels...')
    labels_file_stream = '\n'.join(labels_list)
    labels_file_stream += '\n'
    with open(labels_file_fn, 'w') as f:
        f.write(labels_file_stream)

    print ('writing output node1...')
    node1s_file_stream = '\n'.join(node1s_list)
    node1s_file_stream += '\n'
    with open(node1s_file_fn, 'w') as f:
        f.write(node1s_file_stream)

    print ('writing output node2...')
    node2s_file_stream = '\n'.join(node2s_list)
    node2s_file_stream += '\n'
    with open(node2s_file_fn, 'w') as f:
        f.write(node2s_file_stream)
    
    
    # debug
    # print srl_df[0].shape
    # print srl_df[0]
    # print parse_df[0].shape
    # print parse_df[0]
    # print ct.df2malt(parse_df[1])

    # print 'creating dat...'
    # result_fname = os.path.join(os.getcwd(), plain_xml_fname)
    # with open(result_fname, 'w+b') as of:
    #     of.write(plain_output)
    # result_fname = os.path.join(os.getcwd(), unique_xml_fname)
    # with open(result_fname, 'w+b') as of:
    #     of.write(unique_pairs)
