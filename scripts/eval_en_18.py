""""Evaluates generated sentences.

Please, install NLTK: https://www.nltk.org/install.html
sudo pip install -U nltk


python eval.py <system-dir> <reference-dir>

e.g.
python bin/eval.py system_out_dev Finall4/Sentences/dev/

Author: Bernd Bohnet, bohnetbd@gmail.com, Simon Mille, simon.mille@upf.edu
"""

import codecs
import collections
import io
import os
import sys
import glob
import re

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

def main():
  arguments = sys.argv[1:]
  num_args = len(arguments)
  if num_args != 2:
    print('Wrong number few arguments.')
    print(str(sys.argv[0]), 'system-dir', 'reference-dir')
    exit()
  system_path = arguments[0]
  ref_path = arguments[1]

  output_file_list = os.listdir(system_path)
  # For all files in system path.
  for filename in sorted(output_file_list):
    if filename[0] == '.':
        continue
    print('Filename', str(filename))
    system_filename = os.path.join(system_path, filename)
    corpus_names = re.findall(filename, r'.*delex-(.*dev_DEEP)\.txt')
    if corpus_names:
        corpus_name = corpus_names[-1]
    else:
        corpus_name = 'en_ewt-ud-dev_DEEP'
    ref_filename = os.path.join(ref_path, corpus_name + '.output.dat')
    print('corpus_name', str(corpus_name))

    ref = read_corpus_lines(ref_filename, ref=True)
    hyp = read_corpus_lines(system_filename, ref=False)

    # NIST score
    nist = ns.corpus_nist(ref, hyp, n=4)

    # BLEU score
    chencherry = bs.SmoothingFunction()
    bleu = bs.corpus_bleu(ref, hyp, smoothing_function=chencherry.method2)
    total_len = 0.0
    edi = 0.0
    for r, h in zip(ref, hyp):
      total_len += max(len(r[0]), len(h))
      edi += edit_distance(r[0], h)
    dist = 1-edi/total_len
    
    print('BLEU\tNIST\tDIST')
    print(str(round(bleu*100, 2)) + '\t' + str(round(nist, 3)) + '\t' + str(round(dist*100,2)))
    print()


if __name__ == "__main__":
    main()
