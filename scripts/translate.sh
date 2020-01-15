# generate.sh

# SR 19 T2 current best GCN on test data
nohup python3 translate.py \
-src processed_corpus/SR19_T2_delex_test/test_T2-test_en_ewt-ud-test_DEEP.delex-src-nodes.txt \
-tgt processed_corpus/SR19_T2_delex_test/test_T2-test_en_ewt-ud-test_DEEP.output.dat \
-src_label processed_corpus/SR19_T2_delex_test/test_T2-test_en_ewt-ud-test_DEEP.delex-src-labels.txt \
-src_node1 processed_corpus/SR19_T2_delex_test/test_T2-test_en_ewt-ud-test_DEEP.delex-src-node1.txt \
-src_node2 processed_corpus/SR19_T2_delex_test/test_T2-test_en_ewt-ud-test_DEEP.delex-src-node2.txt \
-replace_unk \
-verbose \
-dynamic_dict \
-batch_size 1 \
-max_length 5 \
-block_ngram_repeat 3 \
-model save/SR19_T2_GCN_4L_reuse_AND_copy_attn/SR19_T2_GCN_4L_reuse_AND_copy_attn_acc_50.26_ppl_18.29_e12.pt \
-output save/SR19_T2_GCN_4L_reuse_AND_copy_attn/SR19_T2_GCN_4L_reuse_AND_copy_attn_delex_predictions_test_block3.txt \
-data_type gcn > eval_SR18_T1_GCN_reuse_copy_attn.out & 



-share_vocab \
-report_bleu \
-max_length 30 \



# T2 current best GCN
nohup python3 translate.py \
-src processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-test.delex-src-nodes.txt \
-tgt processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-test.output.dat \
-src_label processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-test.delex-src-labels.txt \
-src_node1 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-test.delex-src-node1.txt \
-src_node2 processed_corpus/SR18_T2_delexicalsied_en/T2_en-ud-test.delex-src-node2.txt \
-replace_unk \
-verbose \
-dynamic_dict \
-batch_size 1 \
-max_length 5 \
-block_ngram_repeat 3 \
-gpu 3 \
-model save/SR18_T2_GCN_reuse_AND_copy_attn/SR18_T2_GCN_reuse_AND_copy_attn_acc_49.36_ppl_18.46_e11.pt \
-output save/SR18_T2_GCN_reuse_AND_copy_attn/SR18_T2_GCN_reuse_AND_copy_attn_delex_predictions_test.txt \
-data_type gcn > SR18_T2_GCN_reuse_AND_copy_attn.out & 



# current best seq
nohup python3 translate.py \
-src processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-test.delex-src-nodes.txt \
-tgt processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-test.output.dat \
-src_label processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-test.delex-src-labels.txt \
-src_node1 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-test.delex-src-node1.txt \
-src_node2 processed_corpus/SR18_T1_delexicalsied_en/T1_en-ud-test.delex-src-node2.txt \
-replace_unk \
-verbose \
-report_bleu \
-dynamic_dict \
-gpu 1 \
-model save/SR18_T1_BRNN/SR18_T1_BRNN_acc_40.07_ppl_112.33_e15.pt \
-output save/SR18_T1_BRNN/SR18_T1_BRNN_delex_predictions_test.txt \
-data_type text > eval_SR18_T1_BRNN.out & 

