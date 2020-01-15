# train.sh
# SR19_T1_dyn


# EN all 
nohup python3 train.py \
-data data/SR19_T2_GCN_en_all/SR19_T2_GCN_en_all \
-save_model save/SR19_T2_en_all_4L/SR19_T2_en_all_4L \
-rnn_size 512 \
-word_vec_size 512 \
-layers 2 \
-epochs 30 \
-optim adam \
-learning_rate 0.001 \
-dropout 0.5 \
-encoder_type gcn \
-gcn_num_inputs 512 \
-gcn_num_units 512 \
-gcn_in_arcs \
-gcn_out_arcs \
-gcn_num_layers 4 \
-gcn_num_labels 19 \
-gcn_residual residual \
-copy_attn \
-reuse_copy_attn \
-gpuid 0 \
> SR19_T2_en_all_4L.out &


# ES all 
nohup python3 train.py \
-data data/SR19_T2_GCN_es_all/SR19_T2_GCN_es_all \
-save_model save/SR19_T2_es_all_4L/SR19_T2_es_all_4L \
-rnn_size 512 \
-word_vec_size 512 \
-layers 2 \
-epochs 30 \
-optim adam \
-learning_rate 0.001 \
-dropout 0.5 \
-encoder_type gcn \
-gcn_num_inputs 512 \
-gcn_num_units 512 \
-gcn_in_arcs \
-gcn_out_arcs \
-gcn_num_layers 4 \
-gcn_num_labels 16 \
-gcn_residual residual \
-copy_attn \
-reuse_copy_attn \
-gpuid 1 \
> SR19_T2_es_all_4L.out &


# FR partut (all)
nohup python3 train.py \
-data data/SR19_T2_GCN_fr_partut/SR19_T2_GCN_fr_partut \
-save_model save/SR19_T2_fr_all_2L/SR19_T2_fr_all_2L \
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
-gcn_num_labels 18 \
-gcn_residual residual \
-copy_attn \
-reuse_copy_attn \
-gpuid 2 \
> SR19_T2_fr_all_2L.out &





########## current best GCN
nohup python3 train.py \
-data data/SR19_T2_GCN_en_ewt/SR19_T2_GCN_en_ewt \
-save_model save/SR19_T2_GCN_reuse_AND_copy_attn/SR19_T2_GCN_reuse_AND_copy_attn \
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
-gcn_num_layers 4 \
-gcn_num_labels 18 \
-gcn_residual residual \
-copy_attn \
-reuse_copy_attn \
-gpuid 4 \
> SR19_T2_GCN_reuse_AND_copy_attn.out &



# current best seq
nohup python3 train.py \
-data data/SR19_T2_seq_dyn/SR19_T2_seq_dyn \
-save_model save/SR19_T2_RNN/SR19_T2_RNN \
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
> SR19_T2_RNN.out &

-encoder_type rnn \
-gpuid 0 \
> SR19_T2_RNN.out &

-encoder_type brnn \
-gpuid 1 \
> SR19_T2_BRNN.out &

-encoder_type transformer \
-gpuid 2 \
> SR19_T2_Transformer.out &

-encoder_type cnn \
-gpuid 3 \
> SR19_T2_CNN.out &




# coverage + reuse copy_attn
nohup python3 train.py \
-data data/SR19_T1_GCN_dyn/SR19_T1_GCN_dyn \
-save_model save/SR19_T1_GCN_reuse_AND_coverage_attn_lambda/SR19_T1_GCN_reuse_AND_coverage_attn_lambda \
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
> SR19_T1_GCN_reuse_AND_coverage_attn_lambda.out &


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

