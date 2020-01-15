# preprocess.sh

################ GCN
# SR19_T2_dyn
# w/ dynamic vocabularies

# EN all 
python3 preprocess.py \
-train_src processed_corpus/SR19_T2_delex_en_all/en_all-ud-train_DEEP.delex-src-nodes.txt \
-train_label processed_corpus/SR19_T2_delex_en_all/en_all-ud-train_DEEP.delex-src-labels.txt \
-train_node1 processed_corpus/SR19_T2_delex_en_all/en_all-ud-train_DEEP.delex-src-node1.txt \
-train_node2 processed_corpus/SR19_T2_delex_en_all/en_all-ud-train_DEEP.delex-src-node2.txt \
-train_tgt processed_corpus/SR19_T2_delex_en_all/en_all-ud-train_DEEP.output.dat \
-valid_src processed_corpus/SR19_T2_delex_en/T2-dev_en_ewt-ud-dev_DEEP.delex-src-nodes.txt \
-valid_label processed_corpus/SR19_T2_delex_en/T2-dev_en_ewt-ud-dev_DEEP.delex-src-labels.txt \
-valid_node1 processed_corpus/SR19_T2_delex_en/T2-dev_en_ewt-ud-dev_DEEP.delex-src-node1.txt \
-valid_node2 processed_corpus/SR19_T2_delex_en/T2-dev_en_ewt-ud-dev_DEEP.delex-src-node2.txt \
-valid_tgt processed_corpus/SR19_T2_delex_en/T2-dev_en_ewt-ud-dev_DEEP.output.dat \
-save_data data/SR19_T2_GCN_en_all/SR19_T2_GCN_en_all -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type gcn -dynamic_dict

# ES all 
python3 preprocess.py \
-train_src processed_corpus/SR19_T2_delex_es_all/es_all-ud-train_DEEP.delex-src-nodes.txt \
-train_label processed_corpus/SR19_T2_delex_es_all/es_all-ud-train_DEEP.delex-src-labels.txt \
-train_node1 processed_corpus/SR19_T2_delex_es_all/es_all-ud-train_DEEP.delex-src-node1.txt \
-train_node2 processed_corpus/SR19_T2_delex_es_all/es_all-ud-train_DEEP.delex-src-node2.txt \
-train_tgt processed_corpus/SR19_T2_delex_es_all/es_all-ud-train_DEEP.output.dat \
-valid_src processed_corpus/SR19_T2_delex_es/T2-dev_es_ancora-ud-dev_DEEP.delex-src-nodes.txt \
-valid_label processed_corpus/SR19_T2_delex_es/T2-dev_es_ancora-ud-dev_DEEP.delex-src-labels.txt \
-valid_node1 processed_corpus/SR19_T2_delex_es/T2-dev_es_ancora-ud-dev_DEEP.delex-src-node1.txt \
-valid_node2 processed_corpus/SR19_T2_delex_es/T2-dev_es_ancora-ud-dev_DEEP.delex-src-node2.txt \
-valid_tgt processed_corpus/SR19_T2_delex_es/T2-dev_es_ancora-ud-dev_DEEP.output.dat \
-save_data data/SR19_T2_GCN_es_all/SR19_T2_GCN_es_all -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type gcn -dynamic_dict


# FR partut (all)
python3 preprocess.py \
-train_src processed_corpus/SR19_T2_delex_fr/T2-train_fr_partut-ud-train_DEEP.delex-src-nodes.txt \
-train_label processed_corpus/SR19_T2_delex_fr/T2-train_fr_partut-ud-train_DEEP.delex-src-labels.txt \
-train_node1 processed_corpus/SR19_T2_delex_fr/T2-train_fr_partut-ud-train_DEEP.delex-src-node1.txt \
-train_node2 processed_corpus/SR19_T2_delex_fr/T2-train_fr_partut-ud-train_DEEP.delex-src-node2.txt \
-train_tgt processed_corpus/SR19_T2_delex_fr/T2-train_fr_partut-ud-train_DEEP.output.dat \
-valid_src processed_corpus/SR19_T2_delex_fr/T2-dev_fr_partut-ud-dev_DEEP.delex-src-nodes.txt \
-valid_label processed_corpus/SR19_T2_delex_fr/T2-dev_fr_partut-ud-dev_DEEP.delex-src-labels.txt \
-valid_node1 processed_corpus/SR19_T2_delex_fr/T2-dev_fr_partut-ud-dev_DEEP.delex-src-node1.txt \
-valid_node2 processed_corpus/SR19_T2_delex_fr/T2-dev_fr_partut-ud-dev_DEEP.delex-src-node2.txt \
-valid_tgt processed_corpus/SR19_T2_delex_fr/T2-dev_fr_partut-ud-dev_DEEP.output.dat \
-save_data data/SR19_T2_GCN_fr_partut/SR19_T2_GCN_fr_partut -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type gcn -dynamic_dict






export glove_dir="../embeddings"
python3 tools/embeddings_to_torch.py \
    -emb_file "$glove_dir/glove.6B.200d.txt" \
    -dict_file "data/SR19_T2_GCN/SR19_T2_GCN.vocab.pt" \
    -output_file "data/SR19_T2_GCN/SR19_T2_GCN.embeddings" 



# SR19_T1_dyn
# w/ dynamic vocabularies
python3 preprocess.py \
-train_src processed_corpus/SR19_T1_delex_en/T1_en-ud-train.delex-src-nodes.txt \
-train_label processed_corpus/SR19_T1_delex_en/T1_en-ud-train.delex-src-labels.txt \
-train_node1 processed_corpus/SR19_T1_delex_en/T1_en-ud-train.delex-src-node1.txt \
-train_node2 processed_corpus/SR19_T1_delex_en/T1_en-ud-train.delex-src-node2.txt \
-train_tgt processed_corpus/SR19_T1_delex_en/T1_en-ud-train.output.dat \
-valid_src processed_corpus/SR19_T1_delex_en/T1_en-ud-dev.delex-src-nodes.txt \
-valid_label processed_corpus/SR19_T1_delex_en/T1_en-ud-dev.delex-src-labels.txt \
-valid_node1 processed_corpus/SR19_T1_delex_en/T1_en-ud-dev.delex-src-node1.txt \
-valid_node2 processed_corpus/SR19_T1_delex_en/T1_en-ud-dev.delex-src-node2.txt \
-valid_tgt processed_corpus/SR19_T1_delex_en/T1_en-ud-dev.output.dat \
-save_data data/SR19_T1_GCN_dyn/SR19_T1_GCN_dyn -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type gcn -dynamic_dict \
-share_vocab


export glove_dir="../embeddings"
python3 tools/embeddings_to_torch.py \
    -emb_file "$glove_dir/glove.6B.200d.txt" \
    -dict_file "data/SR19_T1_GCN_dyn/SR19_T1_GCN_dyn.vocab.pt" \
    -output_file "data/SR19_T1_GCN_dyn/SR19_T1_GCN_dyn.embeddings" 















################ Seq
# SR19_T2 dyn
python3 preprocess.py \
-train_src processed_corpus/SR19_T2_delex_en/T2-train_en_ewt-ud-train_DEEP.delex-src-nodes.txt \
-train_label processed_corpus/SR19_T2_delex_en/T2-train_en_ewt-ud-train_DEEP.delex-src-labels.txt \
-train_node1 processed_corpus/SR19_T2_delex_en/T2-train_en_ewt-ud-train_DEEP.delex-src-node1.txt \
-train_node2 processed_corpus/SR19_T2_delex_en/T2-train_en_ewt-ud-train_DEEP.delex-src-node2.txt \
-train_tgt processed_corpus/SR19_T2_delex_en/T2-train_en_ewt-ud-train_DEEP.output.dat \
-valid_src processed_corpus/SR19_T2_delex_en/T2-dev_en_ewt-ud-dev_DEEP.delex-src-nodes.txt \
-valid_label processed_corpus/SR19_T2_delex_en/T2-dev_en_ewt-ud-dev_DEEP.delex-src-labels.txt \
-valid_node1 processed_corpus/SR19_T2_delex_en/T2-dev_en_ewt-ud-dev_DEEP.delex-src-node1.txt \
-valid_node2 processed_corpus/SR19_T2_delex_en/T2-dev_en_ewt-ud-dev_DEEP.delex-src-node2.txt \
-valid_tgt processed_corpus/SR19_T2_delex_en/T2-dev_en_ewt-ud-dev_DEEP.output.dat \
-save_data data/SR19_T2_seq_dyn/SR19_T2_seq_dyn -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type text -dynamic_dict

export glove_dir="../embeddings"
python3 tools/embeddings_to_torch.py \
    -emb_file "$glove_dir/glove.6B.200d.txt" \
    -dict_file "data/SR19_T2_seq_dyn/SR19_T2_seq_dyn.vocab.pt" \
    -output_file "data/SR19_T2_seq_dyn/SR19_T2_seq_dyn.embeddings" 



# SR19_T1_dyn
# w/ dynamic vocabularies
python3 preprocess.py \
-train_src processed_corpus/SR19_T1_delex_en/T1_en-ud-train.delex-src-nodes.txt \
-train_label processed_corpus/SR19_T1_delex_en/T1_en-ud-train.delex-src-labels.txt \
-train_node1 processed_corpus/SR19_T1_delex_en/T1_en-ud-train.delex-src-node1.txt \
-train_node2 processed_corpus/SR19_T1_delex_en/T1_en-ud-train.delex-src-node2.txt \
-train_tgt processed_corpus/SR19_T1_delex_en/T1_en-ud-train.output.dat \
-valid_src processed_corpus/SR19_T1_delex_en/T1_en-ud-dev.delex-src-nodes.txt \
-valid_label processed_corpus/SR19_T1_delex_en/T1_en-ud-dev.delex-src-labels.txt \
-valid_node1 processed_corpus/SR19_T1_delex_en/T1_en-ud-dev.delex-src-node1.txt \
-valid_node2 processed_corpus/SR19_T1_delex_en/T1_en-ud-dev.delex-src-node2.txt \
-valid_tgt processed_corpus/SR19_T1_delex_en/T1_en-ud-dev.output.dat \
-save_data data/SR19_T1_seq_dyn/SR19_T1_seq_dyn -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type text -dynamic_dict \
-share_vocab


export glove_dir="../embeddings"
python3 tools/embeddings_to_torch.py \
    -emb_file "$glove_dir/glove.6B.200d.txt" \
    -dict_file "data/SR19_T1_seq_dyn/SR19_T1_seq_dyn.vocab.pt" \
    -output_file "data/SR19_T1_seq_dyn/SR19_T1_seq_dyn.embeddings" 










################ GCN   property
# SR19_T2_dyn_property
# w/ dynamic vocabularies
# add properties to graph
python3 preprocess.py \
-train_src processed_corpus/SR19_T2_delex_en_property/T2-train_en_ewt-ud-train_DEEP.delex-src-nodes.txt \
-train_label processed_corpus/SR19_T2_delex_en_property/T2-train_en_ewt-ud-train_DEEP.delex-src-labels.txt \
-train_node1 processed_corpus/SR19_T2_delex_en_property/T2-train_en_ewt-ud-train_DEEP.delex-src-node1.txt \
-train_node2 processed_corpus/SR19_T2_delex_en_property/T2-train_en_ewt-ud-train_DEEP.delex-src-node2.txt \
-train_tgt processed_corpus/SR19_T2_delex_en_property/T2-train_en_ewt-ud-train_DEEP.output.dat \
-valid_src processed_corpus/SR19_T2_delex_en_property/T2-dev_en_ewt-ud-dev_DEEP.delex-src-nodes.txt \
-valid_label processed_corpus/SR19_T2_delex_en_property/T2-dev_en_ewt-ud-dev_DEEP.delex-src-labels.txt \
-valid_node1 processed_corpus/SR19_T2_delex_en_property/T2-dev_en_ewt-ud-dev_DEEP.delex-src-node1.txt \
-valid_node2 processed_corpus/SR19_T2_delex_en_property/T2-dev_en_ewt-ud-dev_DEEP.delex-src-node2.txt \
-valid_tgt processed_corpus/SR19_T2_delex_en_property/T2-dev_en_ewt-ud-dev_DEEP.output.dat \
-save_data data/SR19_T2_GCN_dyn_prop/SR19_T2_GCN_dyn_prop -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type gcn -dynamic_dict



export glove_dir="../embeddings"
python3 tools/embeddings_to_torch.py \
    -emb_file "$glove_dir/glove.6B.200d.txt" \
    -dict_file "data/SR19_T2_GCN_dyn_prop/SR19_T2_GCN_dyn_prop.vocab.pt" \
    -output_file "data/SR19_T2_GCN_dyn_prop/SR19_T2_GCN_dyn_prop.embeddings" 



# SR19_T1_dyn
# w/ dynamic vocabularies
# add properties to graph
python3 preprocess.py \
-train_src processed_corpus/SR19_T1_delex_en_property/T1_en-ud-train.delex-src-nodes.txt \
-train_label processed_corpus/SR19_T1_delex_en_property/T1_en-ud-train.delex-src-labels.txt \
-train_node1 processed_corpus/SR19_T1_delex_en_property/T1_en-ud-train.delex-src-node1.txt \
-train_node2 processed_corpus/SR19_T1_delex_en_property/T1_en-ud-train.delex-src-node2.txt \
-train_tgt processed_corpus/SR19_T1_delex_en_property/T1_en-ud-train.output.dat \
-valid_src processed_corpus/SR19_T1_delex_en_property/T1_en-ud-dev.delex-src-nodes.txt \
-valid_label processed_corpus/SR19_T1_delex_en_property/T1_en-ud-dev.delex-src-labels.txt \
-valid_node1 processed_corpus/SR19_T1_delex_en_property/T1_en-ud-dev.delex-src-node1.txt \
-valid_node2 processed_corpus/SR19_T1_delex_en_property/T1_en-ud-dev.delex-src-node2.txt \
-valid_tgt processed_corpus/SR19_T1_delex_en_property/T1_en-ud-dev.output.dat \
-save_data data/SR19_T1_GCN_dyn_prop/SR19_T1_GCN_dyn_prop -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type gcn -dynamic_dict 


export glove_dir="../embeddings"
python3 tools/embeddings_to_torch.py \
    -emb_file "$glove_dir/glove.6B.200d.txt" \
    -dict_file "data/SR19_T1_GCN_dyn_prop/SR19_T1_GCN_dyn_prop.vocab.pt" \
    -output_file "data/SR19_T1_GCN_dyn_prop/SR19_T1_GCN_dyn_prop.embeddings" 


