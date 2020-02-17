# train.sh
# SR18_T1_dyn

# current best GCN
nohup python3 train.py \
-data data/SR18_T2_GCN_dyn/SR18_T2_GCN_dyn \
-save_model save/SR18_T2_GCN_reuse_AND_copy_attn/SR18_T2_GCN_reuse_AND_copy_attn \
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
-gcn_num_labels 16 \
-gcn_residual residual \
-copy_attn \
-reuse_copy_attn \
-gpuid 2 \
> SR18_T2_GCN_reuse_AND_copy_attn.out &



# current best seq
nohup python3 train.py \
-data data/SR18_T2_seq_dyn/SR18_T2_seq_dyn \
-save_model save/SR18_T2_RNN/SR18_T2_RNN \
-rnn_size 512 \
-word_vec_size 512 \
-layers 2 \
-epochs 15 \
-optim adam \
-learning_rate 0.001 \
-dropout 0.5 \
-encoder_type rnn \
-copy_attn \
-reuse_copy_attn \
-gpuid 2 \
> SR18_T2_RNN.out &

-encoder_type rnn \
-gpuid 0 \
> SR18_T2_RNN.out &

-encoder_type brnn \
-gpuid 1 \
> SR18_T2_BRNN.out &

-encoder_type transformer \
-gpuid 2 \
> SR18_T2_Transformer.out &

-encoder_type cnn \
-gpuid 3 \
> SR18_T2_CNN.out &




# coverage + reuse copy_attn
nohup python3 train.py \
-data data/SR18_T1_GCN_dyn/SR18_T1_GCN_dyn \
-save_model save/SR18_T1_GCN_reuse_AND_coverage_attn_lambda/SR18_T1_GCN_reuse_AND_coverage_attn_lambda \
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
-gcn_num_labels 16 \
-gcn_residual residual \
-reuse_copy_attn \
-coverage_attn \
-lambda_coverage 1 \
-gpuid 3 \
> SR18_T1_GCN_reuse_AND_coverage_attn_lambda.out &


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



# SR18_T2
python3 train.py \
-data data/SR18_T2_GCN_dyn/SR18_T2_GCN_dyn \
-save_model save/SR18_T2_GCN_2L_512/SR18_T2_GCN_2L_512 \
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
-gcn_num_layers 2 \
-gcn_num_labels 16 \
-gpuid 5 \








# -pre_word_vecs_enc data/SR18_T2_GCN/SR18_T2_GCN.embeddings.enc.pt \
# -pre_word_vecs_dec data/SR18_T2_GCN/SR18_T2_GCN.embeddings.dec.pt \


