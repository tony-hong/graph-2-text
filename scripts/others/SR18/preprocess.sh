# preprocess.sh


################ GCN
# SR18_T2_dyn_property
# w/ dynamic vocabularies
# add properties to graph
python3 preprocess.py \
-train_src processed_corpus/SR18_T2_delexicalsied_en_property/T2_en-ud-train.delex-src-nodes.txt \
-train_label processed_corpus/SR18_T2_delexicalsied_en_property/T2_en-ud-train.delex-src-labels.txt \
-train_node1 processed_corpus/SR18_T2_delexicalsied_en_property/T2_en-ud-train.delex-src-node1.txt \
-train_node2 processed_corpus/SR18_T2_delexicalsied_en_property/T2_en-ud-train.delex-src-node2.txt \
-train_tgt processed_corpus/SR18_T2_delexicalsied_en_property/T2_en-ud-train.output.dat \
-valid_src processed_corpus/SR18_T2_delexicalsied_en_property/T2_en-ud-dev.delex-src-nodes.txt \
-valid_label processed_corpus/SR18_T2_delexicalsied_en_property/T2_en-ud-dev.delex-src-labels.txt \
-valid_node1 processed_corpus/SR18_T2_delexicalsied_en_property/T2_en-ud-dev.delex-src-node1.txt \
-valid_node2 processed_corpus/SR18_T2_delexicalsied_en_property/T2_en-ud-dev.delex-src-node2.txt \
-valid_tgt processed_corpus/SR18_T2_delexicalsied_en_property/T2_en-ud-dev.output.dat \
-save_data data/SR18_T2_GCN_dyn_prop/SR18_T2_GCN_dyn_prop -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type gcn -dynamic_dict



export glove_dir="../embeddings"
python3 tools/embeddings_to_torch.py \
    -emb_file "$glove_dir/glove.6B.200d.txt" \
    -dict_file "data/SR18_T2_GCN_dyn_prop/SR18_T2_GCN_dyn_prop.vocab.pt" \
    -output_file "data/SR18_T2_GCN_dyn_prop/SR18_T2_GCN_dyn_prop.embeddings" 



# SR18_T1_dyn
# w/ dynamic vocabularies
# add properties to graph
python3 preprocess.py \
-train_src processed_corpus/SR18_T1_delexicalsied_en_property/T1_en-ud-train.delex-src-nodes.txt \
-train_label processed_corpus/SR18_T1_delexicalsied_en_property/T1_en-ud-train.delex-src-labels.txt \
-train_node1 processed_corpus/SR18_T1_delexicalsied_en_property/T1_en-ud-train.delex-src-node1.txt \
-train_node2 processed_corpus/SR18_T1_delexicalsied_en_property/T1_en-ud-train.delex-src-node2.txt \
-train_tgt processed_corpus/SR18_T1_delexicalsied_en_property/T1_en-ud-train.output.dat \
-valid_src processed_corpus/SR18_T1_delexicalsied_en_property/T1_en-ud-dev.delex-src-nodes.txt \
-valid_label processed_corpus/SR18_T1_delexicalsied_en_property/T1_en-ud-dev.delex-src-labels.txt \
-valid_node1 processed_corpus/SR18_T1_delexicalsied_en_property/T1_en-ud-dev.delex-src-node1.txt \
-valid_node2 processed_corpus/SR18_T1_delexicalsied_en_property/T1_en-ud-dev.delex-src-node2.txt \
-valid_tgt processed_corpus/SR18_T1_delexicalsied_en_property/T1_en-ud-dev.output.dat \
-save_data data/SR18_T1_GCN_dyn_prop/SR18_T1_GCN_dyn_prop -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type gcn -dynamic_dict 


export glove_dir="../embeddings"
python3 tools/embeddings_to_torch.py \
    -emb_file "$glove_dir/glove.6B.200d.txt" \
    -dict_file "data/SR18_T1_GCN_dyn_prop/SR18_T1_GCN_dyn_prop.vocab.pt" \
    -output_file "data/SR18_T1_GCN_dyn_prop/SR18_T1_GCN_dyn_prop.embeddings" 




################ GCN
# SR18_T2_dyn
# w/ dynamic vocabularies
python3 preprocess.py \
-train_src processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.delex-src-nodes.txt \
-train_label processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.delex-src-labels.txt \
-train_node1 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.delex-src-node1.txt \
-train_node2 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.delex-src-node2.txt \
-train_tgt processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.output.dat \
-valid_src processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.delex-src-nodes.txt \
-valid_label processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.delex-src-labels.txt \
-valid_node1 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.delex-src-node1.txt \
-valid_node2 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.delex-src-node2.txt \
-valid_tgt processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.output.dat \
-save_data data/SR18_T2_GCN_dyn/SR18_T2_GCN_dyn -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type gcn -dynamic_dict



export glove_dir="../embeddings"
python3 tools/embeddings_to_torch.py \
    -emb_file "$glove_dir/glove.6B.200d.txt" \
    -dict_file "data/SR18_T2_GCN/SR18_T2_GCN.vocab.pt" \
    -output_file "data/SR18_T2_GCN/SR18_T2_GCN.embeddings" 



# SR18_T1_dyn
# w/ dynamic vocabularies
python3 preprocess.py \
-train_src processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.delex-src-nodes.txt \
-train_label processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.delex-src-labels.txt \
-train_node1 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.delex-src-node1.txt \
-train_node2 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.delex-src-node2.txt \
-train_tgt processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.output.dat \
-valid_src processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.delex-src-nodes.txt \
-valid_label processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.delex-src-labels.txt \
-valid_node1 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.delex-src-node1.txt \
-valid_node2 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.delex-src-node2.txt \
-valid_tgt processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.output.dat \
-save_data data/SR18_T1_GCN_dyn/SR18_T1_GCN_dyn -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type gcn -dynamic_dict \
-share_vocab


export glove_dir="../embeddings"
python3 tools/embeddings_to_torch.py \
    -emb_file "$glove_dir/glove.6B.200d.txt" \
    -dict_file "data/SR18_T1_GCN_dyn/SR18_T1_GCN_dyn.vocab.pt" \
    -output_file "data/SR18_T1_GCN_dyn/SR18_T1_GCN_dyn.embeddings" 





################ Seq
# SR18_T2 dyn
python3 preprocess.py \
-train_src processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.delex-src-nodes.txt \
-train_label processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.delex-src-labels.txt \
-train_node1 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.delex-src-node1.txt \
-train_node2 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.delex-src-node2.txt \
-train_tgt processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.output.dat \
-valid_src processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.delex-src-nodes.txt \
-valid_label processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.delex-src-labels.txt \
-valid_node1 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.delex-src-node1.txt \
-valid_node2 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.delex-src-node2.txt \
-valid_tgt processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.output.dat \
-save_data data/SR18_T2_seq_dyn/SR18_T2_seq_dyn -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type text -dynamic_dict

export glove_dir="../embeddings"
python3 tools/embeddings_to_torch.py \
    -emb_file "$glove_dir/glove.6B.200d.txt" \
    -dict_file "data/SR18_T2_seq_dyn/SR18_T2_seq_dyn.vocab.pt" \
    -output_file "data/SR18_T2_seq_dyn/SR18_T2_seq_dyn.embeddings" 



# SR18_T1_dyn
# w/ dynamic vocabularies
python3 preprocess.py \
-train_src processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.delex-src-nodes.txt \
-train_label processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.delex-src-labels.txt \
-train_node1 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.delex-src-node1.txt \
-train_node2 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.delex-src-node2.txt \
-train_tgt processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.output.dat \
-valid_src processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.delex-src-nodes.txt \
-valid_label processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.delex-src-labels.txt \
-valid_node1 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.delex-src-node1.txt \
-valid_node2 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.delex-src-node2.txt \
-valid_tgt processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.output.dat \
-save_data data/SR18_T1_seq_dyn/SR18_T1_seq_dyn -src_vocab_size 50000 -tgt_vocab_size 50000 -data_type text -dynamic_dict \
-share_vocab


export glove_dir="../embeddings"
python3 tools/embeddings_to_torch.py \
    -emb_file "$glove_dir/glove.6B.200d.txt" \
    -dict_file "data/SR18_T1_seq_dyn/SR18_T1_seq_dyn.vocab.pt" \
    -output_file "data/SR18_T1_seq_dyn/SR18_T1_seq_dyn.embeddings" 