#!/usr/bin/bash

set -e
attn_approx='FT'
attn_type='improved-clustered'
feature_map='FAVOR'
n_dims=256
affix='default'
q_cluster_size=256
n_hashes=2
batch_size=49
seed=42
imagenet_category="eskimo_husky"
imagenet_category="great grey owl, great gray owl, Strix nebulosa"

# assert 4096 % q_cluster_size == 0, 'q_cluster_size must divide 4096'

. parse_options.sh

# download pre-trained model
smyrf_cli=''
out="samples_ft_${attn_type}_${affix}"
  
python main.py --seed=$seed \
  --experiment_name=138k --do_sample --bs=$batch_size \
  --q_cluster_size=$q_cluster_size $smyrf_cli \
  --ema --n_hashes=$n_hashes --imagenet_category="$imagenet_category" \
  --attn_approx $attn_approx \
  --attn_type $attn_type \
  --feature_map $feature_map --n_dims $n_dims \
  --out_file "$out"
