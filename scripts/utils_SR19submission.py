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


PROPERTY_MAP = {
    'en' : ['Tense', 'Voice', 'NumType', 'Degree', 'Typo', 'Definite', 'Abbr', 'Person', 'Number', 'Mood', 'Foreign', 'Aspect', 'Reflex', 'PronType', 'Poss', 'Gender', 'lin', 'ClauseType'],
    'es' : ['NumForm', 'Definite', 'Tense', 'Reflex', 'Poss', 'Gender', 'ClauseType', 'NumType', 'Degree', 'Mood', 'Person', 'PronType', 'Number', 'Polite', 'AdpType', 'A3_hasDuplicate', 'lin', 'Foreign', 'Voice', 'Aspect', 'AdvType'], 
    'fr' : ['Tense', 'Voice', 'NumType', 'Degree', 'Definite', 'Aspect', 'Person', 'Mood', 'Number', 'PronType', 'Reflex', 'Poss', 'Gender', 'lin', 'ClauseType']
}

IDX_2_PROPERTY = {0: 'Tense', 1: 'Voice', 2: 'NumType', 3: 'Degree', 4: 'Typo', 5: 'Definite', 6: 'Abbr', 7: 'Person', 8: 'Number', 9: 'Mood', 10: 'Foreign', 11: 'Aspect', 12: 'Reflex', 13: 'PronType', 14: 'Poss', 15: 'Gender', 16: 'lin', 17: 'ClauseType'}

PROPERTY_2_ID = {'Tense': 0, 'Voice': 1, 'NumType': 2, 'Degree': 3, 'Typo': 4, 'Definite': 5, 'Abbr': 6, 'Person': 7, 'Number': 8, 'Mood': 9, 'Foreign': 10, 'Aspect': 11, 'Reflex': 12, 'PronType': 13, 'Poss': 14, 'Gender': 15, 'lin': 16, 'ClauseType': 17}


def lemmatize(word, pos=wn.VERB):
    lemma = wn.morphy(word, pos)
    return lemma if lemma else word


class CoNLLTools:
    """
    Class that represents a collection of static methods required to parse
    ukWaC corpus.
    <Class implementation in order to wrap a mess of various string processing
    functions into a single class>
    """
    @staticmethod
    def is_ascii(s):
        return all(ord(c) < 128 for c in s)


    @staticmethod
    def validate(token):
        '''validate whether the token is a valid word
        '''
        if re.search(r"[^a-zA-Z']+", token):
            return False
        return True


    @staticmethod
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


    @staticmethod
    def gzip_reader(fname):
        """
        Read a .gz archive and return its contents as a string.
        If the file specified is not an archive it attempt to read it
        as a general file.

        Args:
            *fname* (str) -- file path

        Returns:
            *fdata* (str) -- file contents

        """
        try:
            with gzip.open(fname, 'r') as f:
                fdata = f.read()
        except (OSError, IOError):
            with open(fname, 'r', encoding='utf8') as f:
                fdata = f.read()
        # return clean_bad_chars(fdata)
        return fdata


    @staticmethod
    def gzip_xml(fname):
        """
        Read and compress specified file, remove original file.

        Args:
            *fname* (str) -- file name

        """
        with open(fname, 'r') as f:
            fdata = f.read()
        with gzip.open(fname + '.gz', 'w') as gf:
            gf.write(fdata)
            os.remove(fname)
        print (fname + '.gz successfully archived')


    @staticmethod
    def append_to_xml(fname, root):
        """
        Create xml file header, prettify xml structure and write xml
        representation of the sentences using ``\\r\\n`` as a separator.

        <IMPORTANT! Take into account that output file shall contain sentences
        separated by ``\\r\\n``. Head searching will not work otherwise. This
        is an ugly hack for ``<text id></text>`` tags to contain correct
        sentences.>

        Args:
            | *fname* (str) -- file name to write the data to
            | *root* (xml.etree object) -- xml.etree root object

        """
        rxml_header = re.compile(r'<\?xml version="1.0" \?>')
        ugly = et.tostring(root, 'utf-8', method='xml')
        parsed_xml = mdom.parseString(ugly)
        nice_xml = parsed_xml.toprettyxml(indent=" " * 3)
        even_more_nice_xml = rxml_header.sub('', nice_xml)
        with open(fname, 'a') as f:
            f.write(even_more_nice_xml)
            f.write('\r\n')  # delimiter required by head_searcher


    @staticmethod
    def get_dependants(sent, prd_id):
        """
        Retrieve roles for a given governor.
        * Duplicate roles are NOT filtered out here! 

        Args:
            | *sent* (list) -- a list of word, POS-tag, word index and role
            |  tuples:
                ``[('first/JJ/3', ('I-A1',)), ('album/NN/4', ('E-A1',))]``
            | *prd_id* (int) -- index to access correct ukwac column

        Returns:
            | *role_bag* (list) -- a list of dicts where dep role is key and
               words, POS-tags, word indeces are values:
                ``[{'V': 'justify/VB/20'},
                  {'A1': 'a/DT/21 full/JJ/22 enquiry/NN/23'}]``
            | *role_list* (list) -- a list of dep roles

        """
        # rarg = re.compile(r'(?![O])[A-Z0-9\-]+')
        # in case of bad parsing
        try:
            # dep_roles = [(rarg.match(d[1][prd_id]).group(), d[0]) for d in sent
            #              if rarg.match(d[1][prd_id])]
            dep_roles = [(d[1][prd_id], d[0]) for d in sent
                         if d[1][prd_id]]
        except:
            print ("exception!!!")
            dep_roles = [('', 0)]
        role_bag = []
        role_chunk = ()
        role_list = []
        
        for i in iter(dep_roles):
            # Problem: only some PRDs are included in role_bag
            # May 29, 2019: fixed the bug left here
            if re.match(r'\(.*\*\)', i[0]):
                role = re.findall(r"\((.*)\*", i[0])[0]
                role_list.append(role)
                role_chunk = (i[1],)
                role_bag.append({role : ' '.join(role_chunk)})
                continue
            elif re.match(r'\(.*\*', i[0]):
                role = re.findall(r"\((.*)\*", i[0])[0]
                role_list.append(role)
                role_chunk = (i[1],)
                continue
            elif i[0] == '*':
                role_chunk += (i[1],)
                continue
            elif i[0] == '*)':
                role_chunk += (i[1],)
                role_bag.append({role : ' '.join(role_chunk)})
                continue
            else:
                print ("exception case: ", i)
                continue
        # print role_bag
        return role_bag, role_list

    
    def df_DFS_recur_nofeat(self, df, current, current_idx, visited, nodes, edges, parent_list, child_list):
        ''' performs depth-first search on dataframe
        '''
        # DEBUG
        # print (current)
        # print (current_idx)
        assert int(current_idx) - 1 == current
        dep_array = df[:, 6]
        num_leaf = np.argwhere(dep_array==current_idx).shape[0]
        if not num_leaf:
            parent_idx = df[current, 6]
            parent = int(parent_idx) - 1
            lemma = df[current, 1]
            sem_tag = df[current, 7]

            visited.append(current_idx)
            nodes.append(lemma)
            edges.append(sem_tag)
            parent_list.append(parent)
            child_list.append(current)
        else:
            children = np.argwhere(dep_array==current_idx)[:, 0]
            for child in children:
                child_idx = df[child, 0]
                self.df_DFS_recur(df, child, child_idx, visited, nodes, edges, parent_list, child_list)
            
            parent_idx = df[current, 6]
            parent = int(parent_idx) - 1
            lemma = df[current, 1]
            sem_tag = df[current, 7]

            visited.append(current_idx)
            nodes.append(lemma)
            edges.append(sem_tag)
            parent_list.append(parent)
            child_list.append(current)
        return visited, nodes, edges, parent_list, child_list

    
    def df_DFS_recur(self, df, current, current_idx, visited, nodes, edges, feats, parent_list, child_list):
        ''' performs depth-first search on dataframe
        '''
        # DEBUG
        # print (current)
        # print (current_idx)
        assert int(current_idx) - 1 == current
        dep_array = df[:, 6]
        num_leaf = np.argwhere(dep_array==current_idx).shape[0]
        if not num_leaf:
            parent_idx = df[current, 6]
            parent = int(parent_idx) - 1
            lemma = df[current, 1]
            feat =  df[current, 4]
            sem_tag = df[current, 7]

            visited.append(current_idx)
            nodes.append(lemma)
            edges.append(sem_tag)
            feats.append(feat)
            parent_list.append(parent)
            child_list.append(current)
        else:
            children = np.argwhere(dep_array==current_idx)[:, 0]
            for child in children:
                child_idx = df[child, 0]
                self.df_DFS_recur(df, child, child_idx, visited, nodes, edges, feats, parent_list, child_list)
            
            parent_idx = df[current, 6]
            parent = int(parent_idx) - 1
            lemma = df[current, 1]
            feat =  df[current, 4]
            sem_tag = df[current, 7]

            visited.append(current_idx)
            nodes.append(lemma)
            edges.append(sem_tag)
            feats.append(feat)
            parent_list.append(parent)
            child_list.append(current)
        return visited, nodes, edges, feats, parent_list, child_list

    
    @staticmethod
    def extract_dataframe(conll_data, toLower=False):
        """
        Extract columns from conll files, create an ordered dict of
        ("word", "lemma") pairs and construct sentences for SENNA input.
        
        Be careful! The length of each row in df is 60 by default

        Args:
            *data* (str) -- file contents

        Returns:
            | *dataframe* (np.array) -- a dataframe of conll data. 
                Each line is a sentence in CoNLL format:
                    Word    POS    Parse    Predicate   Frameset 1, 2, ...

        """
        data_lines = [cl for cl in conll_data.split('\n')]
        # print (data_lines[8])
        # test wsj
        # assert (data_lines[8].strip() == "")
        
        # set default column to 10
        COL_NUM = 10

        line = np.array([], dtype='object')
        sentence_df = np.array([], dtype='object')
        dataframe = []
        sentences = []
        first = True
        for dl in data_lines:
            if dl.strip() != "": 
                # skip comments 
                if dl[0] == '#': 
                    if dl[2:6] == 'text': 
                        sentences.append(dl[9:])
                    continue
                if toLower:
                    dl = dl.lower()
                line = np.array(dl.split(), dtype='object')
                if first:
                    COL_NUM = line.shape[0]
                    sentence_df = line
                    first = False
                else:
#                     print ('before', line.shape)
#                     print ('before', sentence_df.shape)
                    if line.shape[-1] != sentence_df.shape[-1]:
                        correct_dim = min(sentence_df.shape[-1], line.shape[-1])
#                         print ('correct_dim', correct_dim)
                        if sentence_df.ndim == 2:
                            new_sent_list = []
                            for l in range(sentence_df.shape[0]):
                                new_sent_list.append(sentence_df[l][:correct_dim])
                            sentence_df = np.array(new_sent_list, dtype='object')
                        else:
                            sentence_df = sentence_df[:correct_dim]
                        line = line[:correct_dim]
#                     print ('after', line.shape)
#                     print ('after', sentence_df.shape)
                    sentence_df = np.vstack((sentence_df, line))
            else:
                if len(sentence_df) != 0:
                    dataframe.append(sentence_df)
                sentence_df = np.array([], dtype='object')
                line = np.array([], dtype='object')
                first = True
        # exclude those without Props 
        dataframe = np.array([t.reshape(-1, COL_NUM) for t in dataframe], dtype='object')
        # print (np.max([len(l[0]) for l in dataframe]))
        # print (dataframe[0])

        return dataframe, sentences
    
    
    def df_2_dat_DEEP(self, df, output_sent, noOutput, property_vocab=None): 
        result = ''
        imgId = 0
        sent_cnt = 0
        isFirst = True
        
        # save original sentence
        org_sent = output_sent

        idx_list = []
        lemma_list = []
        pos_list = []
        property_list = []
        dep_list = []
        sem_tag_list = []
        input_list = []

        delex_token_list = []
        delex_lemma_list = []

        entityID = 0
        head_ID = 0
        entity_head_ID = 0
        # entity dict: { 'entityID' : [list of lemmas] }
        entity_lemma_dict = defaultdict(list)
        # look up dict from head position to entity placeholder
        head2entity = dict()

        # sentence-based need tokeniser here

        # print (parse_df)
        
        # token_list = parse_df[:, 1].tolist()
        # print (token_list)

        '''
        # handle case with just one dim
        if df.ndim != 2:
            df = df.reshape(1, -1)
            print ('Only one dim !!!')
            return None, None, None, None, None, None, None, None, None
        '''

        # handle case with just one line
        if df.shape[0]==1:
            addition_line = np.array(df[0], dtype='object')
            addition_line[0] = '2'
            addition_line[6] = '1'
            df = np.vstack([df, addition_line])
#             print ('Only one line !!!')
#             print (df)
#             print (df.shape)
#             return None, None, None, None, None, None, None
        
        property_keys = set()
        for i, line in enumerate(df):
            # line = sentence[0]
            # is it in CoNLL format ? 
            # idx starting from 1
            idx = line[0]
            lemma = line[1]
#             assert line[2] == '_'
            pos = line[3]
            # format check for deep-track
            # assert line[4] == '_'
            property_str = line[5]
            dep = line[6]
            sem_tag = line[7]
            
            # map non-ASCII lemma to <UNKNOWN>
            # if not self.is_ascii(lemma):
                # print ("map non-ASCII lemma to <UNKNOWN>: ", lemma)
                # lemma = '<UNKNOWN>'

            lower_lemma = lemma.lower()
            lower_pos = pos.lower()
            lower_sem_tag = sem_tag.lower()
            property_tuples = property_str.split('|')
            property_dict = dict()
            for property_tuple in property_tuples:
                elements = property_tuple.split('=')
                if len(elements) != 2:
                    # print("skip broken properties! ")
                    continue
                key = elements[0]
                value = elements[1]
                property_dict[key] = value
                property_keys.add(key)
            
            # lemma = wnl.lemmatize(lower_token, self.penn2wn(pos))
            
            line[2] = '_'

            idx_list.append(idx)
            lemma_list.append(lemma)
            pos_list.append(pos)
            property_list.append(property_dict)
            dep_list.append(dep)
            sem_tag_list.append(sem_tag)
            
            delex_lemma_list.append(lemma)
            
            # TO TEST: add 2 POS filters; 
            if pos in {'X', 'SYM'}: 
                entityID += 1
                head_ID += 1
                
                entity_placeholder = pos + str(head_ID) + str(entityID)
                entity_lemma_dict[entity_placeholder] = lemma

                head2entity[idx] = entity_placeholder
                
                # delexicalise input 
                # TODO: a potential bug here: two entities with the same lemma
                if not noOutput:
                    output_sent = output_sent.replace(lemma, entity_placeholder)

                # delexicalise output
                lemma = entity_placeholder.lower()
                delex_lemma_list.pop()
                delex_lemma_list.append(lemma)
                
            # replace name entities
            # TO TEST: using pos directly instead of ent
            elif pos in {'PROPN', 'NUM'}: 
                entityID += 1

                # this is the head
                if sem_tag != 'NAME':
                    head_ID += 1
                # this is not the head
                else:
                    pass

                entity_placeholder = pos + str(head_ID) + str(entityID)
                entity_lemma_dict[entity_placeholder] = lemma

                head2entity[idx] = entity_placeholder

                # delexicalise input (token-based)
                # entity_idx = token_list.index(lemma)
                # token_list[entity_idx] = entity_placeholder

                # delexicalise input 
                # TODO: a potential bug here: two entities with the same lemma
                if not noOutput:
                    output_sent = output_sent.replace(lemma, entity_placeholder)

                # delexicalise output
                lemma = entity_placeholder.lower()
                delex_lemma_list.pop()
                delex_lemma_list.append(lemma)
                
            triple = ':'.join([pos, sem_tag, dep, lemma])
            # triple = ':'.join([pos, sem_tag, lemma])
            input_list.append(triple)
            
        # print (df)
        # print (dep_list)

        # add properties
        NULL_placeholder = '*'
        lemma_feat_seperator = '||'
        property_seperator = u'￨'
        if property_vocab: 
            delex_node_list = []
            delex_feat_list = []
            for idx, delex_lemma in enumerate(delex_lemma_list):
                property_str_list = []
                for property_key in property_vocab: 
                    property_str = property_list[idx].get(property_key, NULL_placeholder)
                    property_str_list.append(property_str)
                property_postfix = property_seperator.join(property_str_list)
                
                #add POS
                pos = pos_list[idx]
                node_str = delex_lemma + property_seperator + property_postfix + property_seperator + pos
                
                #node_str = delex_lemma + property_seperator + property_postfix
                
                delex_node_list.append(node_str)
                delex_feat_list.append(property_postfix)
            delex_lemma_list = delex_node_list
            delex_feat_array = np.array(delex_feat_list, dtype='object')
            df[:, 4] = delex_feat_array

        # assigning list to np.array: be careful not to change the length
        # applied delex to dataframe
        delex_lemma_array = np.array(delex_lemma_list, dtype='object')
        df[:, 1] = delex_lemma_array
        
        # DFS for GCN
        if '0' not in dep_list:
            print ("root not in dep_list !!!")
            return None, None, None, None, None, None, None, None, None
        
        root = dep_list.index('0')
        root_idx = idx_list[root]

        visited = []
        nodes = []
        edges = []
        feats = []
        parent_list = []
        child_list = []
        visited, nodes, edges, feats, parent_list, child_list = self.df_DFS_recur(df, root, root_idx, visited, nodes, edges, feats, parent_list, child_list)
        
        edges.pop()
        parent_list.pop()
        child_list.pop()
        node1_list = []
        node2_list = []
        for edge_idx, edge in enumerate(edges, start=0):
            parent = parent_list[edge_idx]
            child = child_list[edge_idx]
            parent_idx = df[parent, 0]
            child_idx = df[child, 0]
            node1_idx = visited.index(parent_idx)
            node2_idx = visited.index(child_idx)
            node1_list.append(str(node1_idx))
            node2_list.append(str(node2_idx))

        # DEBUG
        # print (visited)
        # print (nodes)
        # print (edges)
        # print (parent_list)
        # print (child_list)

        # print (node1_list)
        # print (node2_list)
        # print ('')

        new_line = '\t'.join(input_list)
        # print new_line

        # (token-based)
        # output_sent = ' '.join(token_list)
        
        
        org_tokens = word_tokenize(org_sent)
        org_sent = ' '.join(org_tokens)
        
        tokens = word_tokenize(output_sent)
        output_sent = ' '.join(tokens)
        
        nodes_line = '\t'.join(nodes)
        feats_line = '\t'.join(feats)
        labels_line = '\t'.join(edges)
        node1_line = '\t'.join(node1_list)
        node2_line = '\t'.join(node2_list)
        
        return org_sent, output_sent, entity_lemma_dict, nodes_line, feats_line, labels_line, node1_line, node2_line, property_keys


    
    def __init__(self):
        # number of columns until props
        self.COL_2_PROPS = 1
        self.ROLE_LIST = ['A0', 'A1', 'AM-LOC', 'AM-TMP', 'AM-MNR']
        self.ROLE_FILTER = [
            'AM-DIS', 'AM-MOD', 'AM-NEG', 
            'R-A0', 'R-A1', 'R-A2', 
            'R-AM-TMP', 'R-AM-LOC', 'R-AM-CAU', 'R-AM-MNR', 'R-AM-EXT',
            'C-A0', 'C-A1', 'C-V', ]
        # self.ROLE_FILTER = ['AM-MOD', 'AM-NEG', 'AM-DIS']
        
        
        