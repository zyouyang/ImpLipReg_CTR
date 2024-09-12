# ================================ ML-1M =================================
# ------- AutoInt -------
# python run_recbole.py --dataset=ml-1m --model=AutoInt --learning_rate=1e-3 --n_layers=3 --attention_size=8 --num_heads=2 "--dropout_probs=[0.2, 0.2, 0.2]" "--mlp_hidden_size=[256, 256, 256]" --seed=2020
# python run_recbole.py --dataset=ml-1m --model=AutoInt --learning_rate=1e-3 --n_layers=3 --attention_size=8 --num_heads=2 "--dropout_probs=[0.2, 0.2, 0.2]" "--mlp_hidden_size=[256, 256, 256]" --seed=2020 --multi_cls=5 --multi_ratio=0.5
# python run_recbole.py --dataset=ml-1m --model=AutoInt --learning_rate=1e-3 --n_layers=3 --attention_size=8 --num_heads=2 "--dropout_probs=[0.2, 0.2, 0.2]" "--mlp_hidden_size=[256, 256, 256]" --seed=2021
# python run_recbole.py --dataset=ml-1m --model=AutoInt --learning_rate=1e-3 --n_layers=3 --attention_size=8 --num_heads=2 "--dropout_probs=[0.2, 0.2, 0.2]" "--mlp_hidden_size=[256, 256, 256]" --seed=2021 --multi_cls=5 --multi_ratio=0.5
# python run_recbole.py --dataset=ml-1m --model=AutoInt --learning_rate=1e-3 --n_layers=3 --attention_size=8 --num_heads=2 "--dropout_probs=[0.2, 0.2, 0.2]" "--mlp_hidden_size=[256, 256, 256]" --seed=2022
# python run_recbole.py --dataset=ml-1m --model=AutoInt --learning_rate=1e-3 --n_layers=3 --attention_size=8 --num_heads=2 "--dropout_probs=[0.2, 0.2, 0.2]" "--mlp_hidden_size=[256, 256, 256]" --seed=2022 --multi_cls=5 --multi_ratio=0.5
# python run_recbole.py --dataset=ml-1m --model=AutoInt --learning_rate=1e-3 --n_layers=3 --attention_size=8 --num_heads=2 "--dropout_probs=[0.2, 0.2, 0.2]" "--mlp_hidden_size=[256, 256, 256]" --seed=2023
# python run_recbole.py --dataset=ml-1m --model=AutoInt --learning_rate=1e-3 --n_layers=3 --attention_size=8 --num_heads=2 "--dropout_probs=[0.2, 0.2, 0.2]" "--mlp_hidden_size=[256, 256, 256]" --seed=2023 --multi_cls=5 --multi_ratio=0.5
# python run_recbole.py --dataset=ml-1m --model=AutoInt --learning_rate=1e-3 --n_layers=3 --attention_size=8 --num_heads=2 "--dropout_probs=[0.2, 0.2, 0.2]" "--mlp_hidden_size=[256, 256, 256]" --seed=2024
# python run_recbole.py --dataset=ml-1m --model=AutoInt --learning_rate=1e-3 --n_layers=3 --attention_size=8 --num_heads=2 "--dropout_probs=[0.2, 0.2, 0.2]" "--mlp_hidden_size=[256, 256, 256]" --seed=2024 --multi_cls=5 --multi_ratio=0.5

# ------- EulerNet -------
# python run_recbole.py --dataset=ml-1m --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=5e-4 --seed=2020 --save_dataset=True
# python run_recbole.py --dataset=ml-1m --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=5e-4 --seed=2020 --multi_cls=5 --multi_ratio=0.1
# python run_recbole.py --dataset=ml-1m --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=5e-4 --seed=2021 --save_dataset=True
# python run_recbole.py --dataset=ml-1m --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=5e-4 --seed=2021 --multi_cls=5 --multi_ratio=0.1
# python run_recbole.py --dataset=ml-1m --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=5e-4 --seed=2022 --save_dataset=True
# python run_recbole.py --dataset=ml-1m --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=5e-4 --seed=2022 --multi_cls=5 --multi_ratio=0.1
# python run_recbole.py --dataset=ml-1m --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=5e-4 --seed=2023 --save_dataset=True
# python run_recbole.py --dataset=ml-1m --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=5e-4 --seed=2023 --multi_cls=5 --multi_ratio=0.1
# python run_recbole.py --dataset=ml-1m --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=5e-4 --seed=2024 --save_dataset=True
# python run_recbole.py --dataset=ml-1m --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=5e-4 --seed=2024 --multi_cls=5 --multi_ratio=0.1

# ------- DCNV2 -------
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2020
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.5
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.1
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2021 --multi_cls=5 --multi_ratio=0.1
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2022 --multi_cls=5 --multi_ratio=0.1
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2023 --multi_cls=5 --multi_ratio=0.1
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2024 --multi_cls=5 --multi_ratio=0.1

# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.3
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2021 --multi_cls=5 --multi_ratio=0.3
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2022 --multi_cls=5 --multi_ratio=0.3
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2023 --multi_cls=5 --multi_ratio=0.3
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2024 --multi_cls=5 --multi_ratio=0.3

# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.7
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2021 --multi_cls=5 --multi_ratio=0.7
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2022 --multi_cls=5 --multi_ratio=0.7
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2023 --multi_cls=5 --multi_ratio=0.7
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2024 --multi_cls=5 --multi_ratio=0.7

# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.9
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2021 --multi_cls=5 --multi_ratio=0.9
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2022 --multi_cls=5 --multi_ratio=0.9
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2023 --multi_cls=5 --multi_ratio=0.9
# python run_recbole.py --model=DCNV2 --dataset=ml-1m --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[256, 256, 256]" --structure=parallel --reg_weight=2 --seed=2024 --multi_cls=5 --multi_ratio=0.9


# ------- WideDeep -------
# python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2020
# python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2020 --multi_cls=5 --multi_ratio=0.5

# python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2020 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2021 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2022 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2023 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2024 --multi_cls=5 --multi_ratio=0.1

python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2020 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2021 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2022 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2023 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2024 --multi_cls=5 --multi_ratio=0.3

python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2020 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2021 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2022 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2023 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2024 --multi_cls=5 --multi_ratio=0.7

python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2020 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2021 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2022 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2023 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=WideDeep --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.2 "--mlp_hidden_size=[128, 128, 128]" --seed=2024 --multi_cls=5 --multi_ratio=0.9


# ------- FiGNN -------
# python run_recbole.py --model=FiGNN --dataset=ml-1m --learning_rate=5e-3 --n_layers=2 --num_heads=2 --attention_size=8 --attn_dropout_prob=0.2 --hidden_dropout_prob=0.3 --seed=2020
# python run_recbole.py --model=FiGNN --dataset=ml-1m --learning_rate=5e-3 --n_layers=2 --num_heads=2 --attention_size=8 --attn_dropout_prob=0.2 --hidden_dropout_prob=0.3 --seed=2020 --multi_cls=5 --multi_ratio=0.1
# python run_recbole.py --model=FiGNN --dataset=ml-1m --learning_rate=5e-3 --n_layers=2 --num_heads=2 --attention_size=8 --attn_dropout_prob=0.2 --hidden_dropout_prob=0.3 --seed=2021
# python run_recbole.py --model=FiGNN --dataset=ml-1m --learning_rate=5e-3 --n_layers=2 --num_heads=2 --attention_size=8 --attn_dropout_prob=0.2 --hidden_dropout_prob=0.3 --seed=2021 --multi_cls=5 --multi_ratio=0.1
# python run_recbole.py --model=FiGNN --dataset=ml-1m --learning_rate=5e-3 --n_layers=2 --num_heads=2 --attention_size=8 --attn_dropout_prob=0.2 --hidden_dropout_prob=0.3 --seed=2022
# python run_recbole.py --model=FiGNN --dataset=ml-1m --learning_rate=5e-3 --n_layers=2 --num_heads=2 --attention_size=8 --attn_dropout_prob=0.2 --hidden_dropout_prob=0.3 --seed=2022 --multi_cls=5 --multi_ratio=0.1
# python run_recbole.py --model=FiGNN --dataset=ml-1m --learning_rate=5e-3 --n_layers=2 --num_heads=2 --attention_size=8 --attn_dropout_prob=0.2 --hidden_dropout_prob=0.3 --seed=2023
# python run_recbole.py --model=FiGNN --dataset=ml-1m --learning_rate=5e-3 --n_layers=2 --num_heads=2 --attention_size=8 --attn_dropout_prob=0.2 --hidden_dropout_prob=0.3 --seed=2023 --multi_cls=5 --multi_ratio=0.1
# python run_recbole.py --model=FiGNN --dataset=ml-1m --learning_rate=5e-3 --n_layers=2 --num_heads=2 --attention_size=8 --attn_dropout_prob=0.2 --hidden_dropout_prob=0.3 --seed=2024
# python run_recbole.py --model=FiGNN --dataset=ml-1m --learning_rate=5e-3 --n_layers=2 --num_heads=2 --attention_size=8 --attn_dropout_prob=0.2 --hidden_dropout_prob=0.3 --seed=2024 --multi_cls=5 --multi_ratio=0.1


# ------- NFM -------
# python run_recbole.py --model=NFM --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.5 "--mlp_hidden_size=[512, 512, 512]" --seed=2020
# python run_recbole.py --model=NFM --dataset=ml-1m --learning_rate=5e-3 --dropout_prob=0.5 "--mlp_hidden_size=[512, 512, 512]" --seed=2020 --multi_cls=5 --multi_ratio=0.5

# ------- xDeepFM -------
# python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2020 --multi_cls=5 --multi_ratio=0.5
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2020 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2021 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2022 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2023 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2024 --multi_cls=5 --multi_ratio=0.1

python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2020 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2021 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2022 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2023 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2024 --multi_cls=5 --multi_ratio=0.3

python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2020 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2021 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2022 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2023 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2024 --multi_cls=5 --multi_ratio=0.7

python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2020 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2021 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2022 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2023 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=xDeepFM --dataset=ml-1m --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[200, 200, 200]" --seed=2024 --multi_cls=5 --multi_ratio=0.9


# ================================ Yelp2018 =================================
# ------- DCNV2 -------
# python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2020
# python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.5

python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2021 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2022 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2023 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2024 --multi_cls=5 --multi_ratio=0.1

python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2021 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2022 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2023 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2024 --multi_cls=5 --multi_ratio=0.3

python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2021 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2022 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2023 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2024 --multi_cls=5 --multi_ratio=0.7

python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2021 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2022 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2023 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=DCNV2 --dataset=yelp2018 --cross_layer_num=2 --dropout_prob=0.2 --learning_rate=5e-3 "--mlp_hidden_size=[64, 64, 64]" --structure=parallel --reg_weight=2 --seed=2024 --multi_cls=5 --multi_ratio=0.9

# ------- EulerNet -------
# python run_recbole.py --dataset=yelp2018 --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.3 --apply_norm=False --reg_weight=5e-4 --seed=2020
# python run_recbole.py --dataset=yelp2018 --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.3 --apply_norm=False --reg_weight=5e-4 --seed=2020 --multi_cls=5 --multi_ratio=0.5
# python run_recbole.py --dataset=yelp2018 --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.3 --apply_norm=False --reg_weight=5e-4 --seed=2021
# python run_recbole.py --dataset=yelp2018 --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.3 --apply_norm=False --reg_weight=5e-4 --seed=2021 --multi_cls=5 --multi_ratio=0.5
# python run_recbole.py --dataset=yelp2018 --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.3 --apply_norm=False --reg_weight=5e-4 --seed=2022
# python run_recbole.py --dataset=yelp2018 --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.3 --apply_norm=False --reg_weight=5e-4 --seed=2022 --multi_cls=5 --multi_ratio=0.5
# python run_recbole.py --dataset=yelp2018 --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.3 --apply_norm=False --reg_weight=5e-4 --seed=2023
# python run_recbole.py --dataset=yelp2018 --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.3 --apply_norm=False --reg_weight=5e-4 --seed=2023 --multi_cls=5 --multi_ratio=0.5
# python run_recbole.py --dataset=yelp2018 --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.3 --apply_norm=False --reg_weight=5e-4 --seed=2024
# python run_recbole.py --dataset=yelp2018 --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.3 --apply_norm=False --reg_weight=5e-4 --seed=2024 --multi_cls=5 --multi_ratio=0.5

# ------- AutoInt -------
# python run_recbole.py --dataset=yelp2018 --model=AutoInt --num_heads=2 --n_layers=3 --attention_size=8 --learning_rate=1e-3 "--dropout_probs=[0.0, 0.0, 0.0]" "--mlp_hidden_size=[128, 128, 128]" --seed=2020
# python run_recbole.py --dataset=yelp2018 --model=AutoInt --num_heads=2 --n_layers=3 --attention_size=8 --learning_rate=1e-3 "--dropout_probs=[0.0, 0.0, 0.0]" "--mlp_hidden_size=[128, 128, 128]" --seed=2020 --multi_cls=5 --multi_ratio=0.5

# ------- WideDeep -------
# python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2020
# python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2020 --multi_cls=5 --multi_ratio=0.5

python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2020 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2021 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2022 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2023 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2024 --multi_cls=5 --multi_ratio=0.1

python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2020 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2021 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2022 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2023 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2024 --multi_cls=5 --multi_ratio=0.3

python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2020 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2021 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2022 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2023 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2024 --multi_cls=5 --multi_ratio=0.7

python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2020 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2021 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2022 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2023 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=WideDeep --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[256, 256, 256]" --seed=2024 --multi_cls=5 --multi_ratio=0.9

# xDeepFM
# python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2020 --multi_cls=5 --multi_ratio=0.5

python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2020 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2021 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2022 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2023 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2024 --multi_cls=5 --multi_ratio=0.1

python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2020 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2021 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2022 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2023 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2024 --multi_cls=5 --multi_ratio=0.3

python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2020 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2021 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2022 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2023 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2024 --multi_cls=5 --multi_ratio=0.7

python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2020 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2021 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2022 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2023 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=xDeepFM --dataset=yelp2018 --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[512, 512, 512]" "--cin_layer_size=[100, 100, 100]" --seed=2024 --multi_cls=5 --multi_ratio=0.9

# ================================ amazon-books =================================
# ------- DCNV2 -------
# python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2020
# python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.5

python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2021 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2022 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2023 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2024 --multi_cls=5 --multi_ratio=0.1 

python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2021 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2022 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2023 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2024 --multi_cls=5 --multi_ratio=0.3 

python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2021 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2022 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2023 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2024 --multi_cls=5 --multi_ratio=0.7 

python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2020 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2021 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2022 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2023 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=DCNV2 --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --cross_layer_num=4 --dropout_prob=0.2 --learning_rate=1e-3 "--mlp_hidden_size=[64, 64]" --structure=parallel --reg_weight=2 --seed=2024 --multi_cls=5 --multi_ratio=0.9 

# ------- EulerNet -------
# python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=0 --seed=2020
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=0 --seed=2020 --multi_cls=5 --multi_ratio=0.5
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=0 --seed=2021
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=0 --seed=2021 --multi_cls=5 --multi_ratio=0.5
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=0 --seed=2022
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=0 --seed=2022 --multi_cls=5 --multi_ratio=0.5
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=0 --seed=2023
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=0 --seed=2023 --multi_cls=5 --multi_ratio=0.5
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=0 --seed=2024
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --model=EulerNet --learning_rate=1e-3 "--order_list=[8]" --drop_ex=0.3 --drop_im=0.5 --apply_norm=False --reg_weight=0 --seed=2024 --multi_cls=5 --multi_ratio=0.5

# ------- WideDeep -------
# python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2020
# python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2020 --multi_cls=5 --multi_ratio=0.5

python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2020 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2021 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2022 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2023 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2024 --multi_cls=5 --multi_ratio=0.1

python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2020 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2021 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2022 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2023 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2024 --multi_cls=5 --multi_ratio=0.3

python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2020 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2021 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2022 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2023 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2024 --multi_cls=5 --multi_ratio=0.7

python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2020 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2021 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2022 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2023 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --model=WideDeep --learning_rate=5e-4 --dropout_prob=0.1 "--mlp_hidden_size=[64, 64, 64]" --seed=2024 --multi_cls=5 --multi_ratio=0.9

# ------- xDeepFM -------
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2020 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2021 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2022 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2023 --multi_cls=5 --multi_ratio=0.1
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2024 --multi_cls=5 --multi_ratio=0.1

python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2020 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2021 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2022 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2023 --multi_cls=5 --multi_ratio=0.3
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2024 --multi_cls=5 --multi_ratio=0.3

python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2020 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2021 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2022 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2023 --multi_cls=5 --multi_ratio=0.7
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2024 --multi_cls=5 --multi_ratio=0.7

python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2020.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2020 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2021.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2021 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2022.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2022 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2023.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2023 --multi_cls=5 --multi_ratio=0.9
python run_recbole.py --model=xDeepFM --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset-seed2024.pth --learning_rate=1e-4 --dropout_prob=0.5 "--mlp_hidden_size=[256, 256, 256]" "--cin_layer_size=[200, 200, 200]" --seed=2024 --multi_cls=5 --multi_ratio=0.9

# ------- AutoInt -------
# python run_recbole.py --dataset=amazon-books --dataset_save_path=saved/amazon-books-Dataset.pth --model=AutoInt --num_heads=2 --n_layers=3 --attention_size=8 --learning_rate=1e-3 "--dropout_probs=[0.0, 0.0, 0.0]" "--mlp_hidden_size=[64, 64, 64]" --seed=2020


# =======================================================================
# Check ranking metrics
# =======================================================================

# ================================ ML-1M =================================
# ------- DCNV2 -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_13-12-18.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_13-24-44.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_13-12-18.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_13-13-29.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_13-14-43.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_13-15-39.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_13-16-49.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_13-18-12.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_13-19-08.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_13-20-03.pth

# ------- WideDeep -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_13-29-32.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_13-30-17.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_13-31-09.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_13-31-47.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_13-33-20.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_13-34-10.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_13-35-08.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_13-35-54.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_13-37-00.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_13-37-46.pth

# ------- AutoInt -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-16-54.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-18-17.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-19-48.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-20-59.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-22-32.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-23-42.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-25-07.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-26-05.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-27-41.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-28-46.pth

# ------- EulerNet -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_15-04-06.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_15-07-50.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_15-10-55.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_15-14-27.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_15-17-30.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_15-21-34.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_15-24-44.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_15-28-33.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_15-31-20.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_15-34-34.pth

# ------- FiGNN -------
# seed 2020
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_13-47-44.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_13-48-27.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_13-49-01.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_13-49-48.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_13-50-22.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_13-51-10.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_13-51-51.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_13-52-24.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_13-53-02.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_13-53-40.pth

# # ------- NFM -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_13-58-38.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_13-59-34.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_14-01-01.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_14-45-07.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_14-46-15.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_14-46-58.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_14-47-50.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_14-02-00.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_14-03-22.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_14-04-14.pth

# # ------ xDeepFM -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_16-24-45.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Aug-22-2024_00-48-21.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_17-03-07.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_19-00-28.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_19-30-37.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_19-32-42.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_19-07-52.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_19-10-48.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_19-17-24.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_19-20-49.pth

# ================================ Yelp2018 =================================
# # ------- DCNV2 -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Aug-22-2024_01-39-23.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Aug-22-2024_01-40-53.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_20-30-51.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_20-31-53.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_20-33-01.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_20-34-04.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_20-35-08.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_20-36-10.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_20-37-19.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_20-38-20.pth

# # ------- WideDeep -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Aug-22-2024_01-57-28.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Aug-22-2024_01-59-01.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_20-57-38.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_20-58-45.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_20-59-50.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_21-00-56.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_21-02-04.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_21-03-10.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_21-04-16.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_21-05-22.pth

# # ------- AutoInt -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Aug-22-2024_02-07-40.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Aug-22-2024_02-09-32.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-48-30.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-49-38.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-50-47.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-51-55.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-53-03.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-54-12.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-55-21.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_20-56-29.pth

# # ------- EulerNet -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Aug-22-2024_02-19-22.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Aug-22-2024_02-20-58.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_20-39-31.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_20-40-38.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_20-41-46.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_20-42-54.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_20-44-01.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_20-45-07.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_20-46-16.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_20-47-23.pth

# # ------- FiGNN -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_15-48-24.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_15-49-32.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_21-06-34.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_21-07-42.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_21-08-50.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_21-09-57.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_21-11-06.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_21-12-14.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_21-13-23.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_21-14-30.pth

# # ------- NFM -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_16-04-53.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_16-06-02.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_21-20-53.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_21-21-59.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_21-23-05.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_21-24-08.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_21-25-15.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_21-26-24.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_21-27-32.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_21-28-41.pth

# # ------ xDeepFM -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_19-39-07.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_19-40-40.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_21-29-49.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_21-31-21.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_21-33-03.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_21-34-34.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_21-36-08.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_21-37-40.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_21-39-13.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_21-40-45.pth

# ================================ amazon-books =================================
# # ------- DCNV2 -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_22-10-53.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_22-40-58.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_22-06-01.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_22-42-05.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_22-19-11.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_22-43-14.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_22-26-52.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_22-44-22.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_22-31-10.pth
# python recbole_rank_eval.py --checkpoint=saved/DCNV2-Mar-26-2024_22-45-21.pth

# # ------- AutoInt -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Aug-22-2024_17-58-00.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_23-15-40.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_23-16-45.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_23-17-49.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_23-18-53.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_23-19-57.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_23-21-10.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_23-22-14.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_23-23-19.pth
# python recbole_rank_eval.py --checkpoint=saved/AutoInt-Mar-26-2024_23-24-23.pth

# # ------- WideDeep -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Aug-22-2024_18-07-23.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_22-56-47.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_22-57-50.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_22-58-51.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_22-59-53.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_23-01-02.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_23-02-12.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_23-03-14.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_23-04-24.pth
# python recbole_rank_eval.py --checkpoint=saved/WideDeep-Mar-26-2024_23-05-24.pth

# # ------- EulerNet -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Aug-22-2024_18-13-42.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_22-46-29.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_22-47-36.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_22-48-42.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_22-49-41.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_22-50-45.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_22-52-05.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_22-53-15.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_22-54-43.pth
# python recbole_rank_eval.py --checkpoint=saved/EulerNet-Mar-26-2024_22-55-48.pth

# # ------- FiGNN -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_16-06-45.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_23-06-25.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_23-07-27.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_23-08-25.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_23-09-26.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_23-10-24.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_23-11-32.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_23-12-31.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_23-13-33.pth
# python recbole_rank_eval.py --checkpoint=saved/FiGNN-Mar-26-2024_23-14-31.pth

# # ------- NFM -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_16-32-37.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_20-03-31.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_23-26-33.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_23-27-34.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_23-28-39.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_23-29-38.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_23-30-42.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_23-31-42.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_23-32-47.pth
# python recbole_rank_eval.py --checkpoint=saved/NFM-Mar-26-2024_23-33-47.pth

# # ------ xDeepFM -------
# # seed 2020
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_19-43-41.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_23-34-45.pth
# # seed 2021
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_23-42-05.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_23-46-49.pth
# # seed 2022
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_23-54-10.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-26-2024_23-58-53.pth
# # seed 2023
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-27-2024_00-06-14.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-27-2024_00-10-58.pth
# # seed 2024
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-27-2024_00-18-18.pth
# python recbole_rank_eval.py --checkpoint=saved/xDeepFM-Mar-27-2024_00-23-02.pth
