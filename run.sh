CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    --grad_accumulation_steps 1 \
    --teacher_model_name_or_path path/to/your/model \
    --model_name_or_path path/to/your/model \
    --train_data_file path/to/your/dataset \
    --output_dir logs --experiment_name tinyllama_distill --n_gpus 1 --devices [0] \
    --max_seq_len 1024 --batch_size 1 --num_workers 16
