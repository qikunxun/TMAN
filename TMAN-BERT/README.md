#TMAN-BERT


## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [TensorFlow] (https://tensorflow.google.cn/) > 1.10.0

export BERT_BASE_DIR=/path/to/bert/multilingual_L-12_H-768_A-12
export XNLI_DIR=/path/to/xnli

python run_gan.py \
  --task_name=XNLI \
  --do_train=true \
  --do_eval=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=./xnli_output
