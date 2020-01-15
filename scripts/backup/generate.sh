# generate.sh


# SR18_T2 val
python3 translate.py \
-model save/SR18_T2_GCN_init_emb/SR18_T2_GCN_acc_28.52_ppl_295.37_e16.pt \
-data_type gcn \
-src processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.delex-src-nodes.txt \
-tgt processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.output.dat \
-src_label processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.delex-src-labels.txt \
-src_node1 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.delex-src-node1.txt \
-src_node2 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-dev.delex-src-node2.txt \
-output save/SR18_T2_GCN_init_emb/delexicalized_predictions_dev.txt \
-replace_unk \
-gpu 6 \
-verbose


# SR18_T2 train 
python3 translate.py \
-model save/SR18_T2_GCN_init_emb/SR18_T2_GCN_acc_28.52_ppl_295.37_e16.pt \
-data_type gcn \
-src processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.delex-src-nodes.txt \
-tgt processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.output.dat \
-src_label processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.delex-src-labels.txt \
-src_node1 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.delex-src-node1.txt \
-src_node2 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-train.delex-src-node2.txt \
-output save/SR18_T2_GCN_init_emb/delexicalized_predictions_train.txt \
-replace_unk \
-gpu 7 \
-verbose



# SR18_T1 val
python3 translate.py \
-model save/SR18_T1_GCN_2L_512/SR18_T1_GCN_acc_45.00_ppl_101.04_e15.pt \
-data_type gcn \
-src processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.delex-src-nodes.txt \
-tgt processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.output.dat \
-src_label processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.delex-src-labels.txt \
-src_node1 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.delex-src-node1.txt \
-src_node2 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-dev.delex-src-node2.txt \
-output save/SR18_T1_GCN_2L_512/delexicalized_predictions_dev.txt \
-replace_unk \
-gpu 6 \
-verbose


# SR18_T1 train 
python3 translate.py \
-model save/SR18_T1_GCN_2L_512/SR18_T1_GCN_acc_45.00_ppl_101.04_e15.pt \
-data_type gcn \
-src processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.delex-src-nodes.txt \
-tgt processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.output.dat \
-src_label processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.delex-src-labels.txt \
-src_node1 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.delex-src-node1.txt \
-src_node2 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-train.delex-src-node2.txt \
-output save/SR18_T1_GCN_2L_512/delexicalized_predictions_train.txt \
-replace_unk \
-gpu 7 \
-verbose



# WebNLG
# python3 translate.py \
# -model save/WebNLG_GCN/WebNLG_GCN_acc_70.33_ppl_3.30_e17.pt \
# -data_type gcn \
# -src processed_corpus/webnlg/dev-webnlg-all-delex-src-nodes.txt \
# -tgt processed_corpus/webnlg/dev-webnlg-all-delex-tgt.txt \
# -src_label processed_corpus/webnlg/dev-webnlg-all-delex-src-labels.txt \
# -src_node1 processed_corpus/webnlg/dev-webnlg-all-delex-src-node1.txt \
# -src_node2 processed_corpus/webnlg/dev-webnlg-all-delex-src-node2.txt \
# -output save/WebNLG_GCN/delexicalized_predictions_dev.txt \
# -replace_unk \
# -gpu 6 \
# -verbose


