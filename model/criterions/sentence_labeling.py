# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

from fairseq.modules.dynamic_crf_layer import DynamicCRF as CRF
import logging
@register_criterion('sentence_labeling')
class SentenceLabelingCriterion(FairseqCriterion):

    def __init__(self, task, labeling_head_name, regression_target):
        super().__init__(task)
        self.labeling_head_name = labeling_head_name
        self.regression_target = regression_target
        self.crf_layer = CRF(17, beam_size=5)

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--labeling-head-name',
                            default='sentence_labeling_head',
                            help='name of the labeling head to use')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, 'labeling_heads')
            and self.labeling_head_name in model.labeling_heads
        ), 'model must provide sentence labeling head for --criterion=sentence_labeling'
        (loss, _path_score, path), _ = model(
            **sample['net_input'],
            features_only=True,
            labeling_head_name=self.labeling_head_name,
           kwargs=sample
        )
        # logging.info("loss: {},\npath_score: {},\npath: {}".format(loss, path_score, path))
        targets = model.get_targets(sample, [])
        sample_size = targets.numel()

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }
        if not self.regression_target:
            # preds = logits.argmax(dim=1)
            logging_output['ncorrect'] = (path == targets).sum()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2),  round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2),  round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences,  round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
