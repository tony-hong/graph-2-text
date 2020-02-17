# train.sh
# SR18_T1_dyn

# test current best GCN on dataset properties
nohup python3 train.py \
-data data/SR18_T1_GCN_dyn_prop/SR18_T1_GCN_dyn_prop \
-save_model save/test_dataset_prop/SR18_T1_GCN_reuse_copy_attn_3L/SR18_T1_GCN_reuse_copy_attn_3L \
-rnn_size 512 \
-word_vec_size 512 \
-layers 3 \
-epochs 20 \
-optim adam \
-learning_rate 0.001 \
-dropout 0.5 \
-encoder_type gcn \
-gcn_num_inputs 512 \
-gcn_num_units 512 \
-gcn_in_arcs \
-gcn_out_arcs \
-gcn_num_layers 2 \
-gcn_num_labels 79 \
-gcn_residual residual \
-reuse_copy_attn \
-gpuid 2 \
> test_SR18_T1_GCN_reuse_copy_attn_3L.out &



# current best GCN
nohup python3 train.py \
-data data/SR18_T1_GCN_dyn/SR18_T1_GCN_dyn \
-save_model save/SR18_T1_GCN_reuse_copy_attn/SR18_T1_GCN_reuse_copy_attn \
-rnn_size 512 \
-word_vec_size 512 \
-layers 2 \
-epochs 15 \
-optim adam \
-learning_rate 0.001 \
-dropout 0.5 \
-encoder_type gcn \
-gcn_num_inputs 512 \
-gcn_num_units 512 \
-gcn_in_arcs \
-gcn_out_arcs \
-gcn_num_layers 2 \
-gcn_num_labels 47 \
-gcn_residual residual \
-reuse_copy_attn \
-gpuid 0 \
> SR18_T1_GCN_reuse_copy_attn.out &



# current best seq
nohup python3 train.py \
-data data/SR18_T1_seq_dyn/SR18_T1_seq_dyn \
-save_model save/SR18_T1_CNN/SR18_T1_CNN \
-rnn_size 512 \
-word_vec_size 512 \
-layers 2 \
-epochs 20 \
-optim adam \
-learning_rate 0.001 \
-dropout 0.5 \
-reuse_copy_attn \
-encoder_type cnn \
-gpuid 3 \
> SR18_T1_CNN.out &

-encoder_type rnn \
-gpuid 0 \
> SR18_T1_RNN.out &

-encoder_type brnn \
-gpuid 1 \
> SR18_T1_BRNN.out &

-encoder_type transformer \
-gpuid 2 \
> SR18_T1_Transformer.out &

-encoder_type cnn \
-gpuid 3 \
> SR18_T1_CNN.out &



# res + copy_attn
nohup python3 train.py \
-data data/SR18_T1_GCN_dyn/SR18_T1_GCN_dyn \
-save_model save/SR18_T1_GCN_reuse_AND_copy_attn/SR18_T1_GCN_reuse_AND_copy_attn \
-rnn_size 512 \
-word_vec_size 512 \
-layers 2 \
-epochs 15 \
-optim adam \
-learning_rate 0.001 \
-dropout 0.5 \
-encoder_type gcn \
-gcn_num_inputs 512 \
-gcn_num_units 512 \
-gcn_in_arcs \
-gcn_out_arcs \
-gcn_num_layers 2 \
-gcn_num_labels 47 \
-gcn_residual residual \
-copy_attn \
-reuse_copy_attn \
-gpuid 0 \
> SR18_T1_GCN_reuse_AND_copy_attn.out &


-copy_attn \
-gpuid 0 \
-copy_attn_force \
-gpuid 1 \
-reuse_copy_attn \
-gpuid 2 \
-copy_loss_by_seqlength \
-gpuid 3 \
-coverage_attn \
-gpuid 4 \
-lambda_coverage \
-gpuid 5 \







# SR18_T1
python3 train.py \
-data data/SR18_T1_GCN/SR18_T1_GCN \
-save_model save/SR18_T1_GCN_2L_512/SR18_T1_GCN_2L_512_congate \
-rnn_size 512 \
-layers 1 \
-epochs 20 \
-optim adam \
-learning_rate 0.001 \
-encoder_type gcn \
-gcn_num_inputs 512 \
-gcn_num_units 512 \
-gcn_in_arcs \
-gcn_out_arcs \
-gcn_num_layers 2 \
-gcn_num_labels 47 \
-gpuid 5 \


# -pre_word_vecs_enc data/SR18_T1_GCN/SR18_T1_GCN.embeddings.enc.pt \
# -pre_word_vecs_dec data/SR18_T1_GCN/SR18_T1_GCN.embeddings.dec.pt \



# -pre_word_vecs_enc data/SR18_T1_GCN/SR18_T1_GCN.embeddings.enc.pt \
# -pre_word_vecs_dec data/SR18_T1_GCN/SR18_T1_GCN.embeddings.dec.pt \