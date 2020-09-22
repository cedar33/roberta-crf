TOTAL_NUM_UPDATES=14299    # 10 epochs through IMDB for bsz 32
WARMUP_UPDATES=858     # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
HEAD_NAME=kaggle_ner     # Custom name for the classification head.
NUM_CLASSES=17           # Number of classes for the classification task.
MAX_SENTENCES=32         # Batch size.
ROBERTA_PATH=/home/stark/workdir/language_model/roberta.base/model.pt

CUDA_VISIBLE_DEVICES=2 fairseq-train ./ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_labeling \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch roberta_base \
    --criterion sentence_labeling \
    --labeling-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 20 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --update-freq 4 \
    --tensorboard-logdir "./tensorboard_log"
