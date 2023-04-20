
import torch
import torch.nn.functional as F

import math

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq import utils
from fairseq.logging import metrics
#from fairseq.criterions.nat_loss import LabelSmoothedDualImitationCriterion

from dataclasses import dataclass, field

@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="bleu",
                                       metadata={"help": "sentence level metric"})


@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task)
        self.metric = sentence_level_metric
        self.initialize_sacrebleu()

    def initialize_sacrebleu(self):
        if self.metric == "BLEU":
            from sacrebleu.metrics import BLEU
            self.sacrebleu = BLEU(effective_order="True")
        elif self.metric == "CHRF":
            from sacrebleu.metrics import CHRF
            self.sacrebleu = CHRF()

    def compute_metric(self, hypothesis, reference):
        return self.sacrebleu.corpus_score(hypothesis, reference).score

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """

        logging_output = str() #add things to log here

        #padding mask, do not remove
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        #we take a softmax over outputs
        output_probs = F.softmax(outputs, dim=1)
        
        #argmax over the softmax \ sampling (e.g. multinomial)
        #sampled_sentence = [4, 17, 18, 19, 20]
        sampled_indices = torch.multinomial(output_probs, num_samples=1) # sampling, not argmax (research question?)
        
        #sampled_sentence_string = tgt_dict.string([4, 17, 18, 19, 20])
        #see dictionary class of fairseq
        sampled_text = self.task.target_dictionary.string(sampled_indices)
        
        #target_sentence = "I am a sentence"
        target_text = self.task.target_dictionary.string(targets)

        #with torch.no_grad()
            #R(*) = eval_metric(sampled_sentence_string, target_sentence)
            #R(*) is a number, BLEU, сhrf, etc.

        # we calculate the metric once per batch (there is only one "corpus string" in the batch)
        with torch.no_grad():
          score = self.compute_metric([sampled_text], [[target_text]])
        
        loss = torch.gather(output_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        print(loss.shape) # the shape should be num_tokens x 1 (the number being the loss of a token)
        loss = loss.mean()
        print(loss)

        sample_size = 1

        #import sys
        #sys.exit(0)

        return loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        #get loss only on tokens, not on lengths
        outs = outputs["word_ins"].get("out", None)
        masks = outputs["word_ins"].get("mask", None)

        loss = self._compute_loss(outs, tgt_tokens, masks)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.detach(),
            "nll_loss": loss.detach(),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )