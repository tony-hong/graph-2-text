# train.sh


# python3 train.py \
# -data data/WebNLG_GCN/gcn_exp \
# -save_model save/WebNLG_GCN \
# -rnn_size 256 \
# -word_vec_size 256 \
# -layers 1 \
# -epochs 10 \
# -optim adam \
# -learning_rate 0.001 \
# -encoder_type gcn \
# -gcn_num_inputs 256 \
# -gcn_num_units 256 \
# -gcn_in_arcs \
# -gcn_out_arcs \
# -gcn_num_layers 1 \
# -gcn_num_labels 5


# SR18_T2
python3 train.py \
-data data/SR18_T2_GCN/SR18_T2_GCN \
-save_model save/SR18_T2_GCN_2L_512/SR18_T2_GCN_1L_512 \
-rnn_size 512 \
-word_vec_size 512 \
-layers 1 \
-epochs 20 \
-optim adam \
-learning_rate 0.001 \
-encoder_type gcn \
-gcn_num_inputs 512 \
-gcn_num_units 512 \
-gcn_in_arcs \
-gcn_out_arcs \
-gcn_num_layers 1 \
-gcn_num_labels 16 \
-gpuid 0 \
# -pre_word_vecs_enc data/SR18_T2_GCN/SR18_T2_GCN.embeddings.enc.pt \
# -pre_word_vecs_dec data/SR18_T2_GCN/SR18_T2_GCN.embeddings.dec.pt \




# SR18_T1
python3 train.py \
-data data/SR18_T1_GCN/SR18_T1_GCN \
-save_model save/SR18_T1_GCN_2L_512/SR18_T1_GCN_1L_512 \
-rnn_size 512 \
-word_vec_size 512 \
-layers 1 \
-epochs 20 \
-optim adam \
-learning_rate 0.001 \
-encoder_type gcn \
-gcn_num_inputs 512 \
-gcn_num_units 512 \
-gcn_in_arcs \
-gcn_out_arcs \
-gcn_num_layers 1 \
-gcn_num_labels 47 \
-gpuid 1 \
# -pre_word_vecs_enc data/SR18_T1_GCN/SR18_T1_GCN.embeddings.enc.pt \
# -pre_word_vecs_dec data/SR18_T1_GCN/SR18_T1_GCN.embeddings.dec.pt \