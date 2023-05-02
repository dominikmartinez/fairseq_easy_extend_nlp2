
import torch
import torch.nn.functional as F

import math

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq import utils
from fairseq.logging import metrics
from fairseq.data import encoders

from argparse import Namespace

from dataclasses import dataclass, field

import wandb


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
        self.tokenizer = encoders.build_tokenizer(Namespace(tokenizer="moses"))

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

        def mask_then_sample(outputs, targets, masks):
            # masking 
            if masks is not None:
                outputs, targets = outputs[masks], targets[masks]
            
            # sampling
            output_probs = F.softmax(outputs, dim=1)
            sampled_indices = torch.multinomial(output_probs, num_samples=1)
            
            # get strings
            sampled_string = self.task.target_dictionary.string(sampled_indices, bpe_symbol="subword_nmt")
            sampled_string = self.tokenizer.decode(sampled_string)
            target_string = self.task.target_dictionary.string(targets, bpe_symbol="subword_nmt")
            
            # scoring
            with torch.no_grad():
                reward = self.compute_metric([sampled_string], [[target_string]])
            
            # loss calculation
            log_probs = torch.log(output_probs)
            log_probs_of_samples = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            loss = -log_probs_of_samples * reward
            loss = loss.mean()
            
            # return loss
            return loss, reward

      
        def sample_then_mask(outputs, targets, masks):
            # sampling
            output_probs = F.softmax(outputs)
            B, T, V = output_probs.shape[0], output_probs.shape[1], output_probs.shape[2]
            output_probs_flat = output_probs.view(B*T, V)
            sampled_indices_flat = torch.multinomial(output_probs_flat, num_samples=1)
            sampled_indices = sampled_indices_flat.view(B, T)

            # get strings
            sampled_strings = [self.task.target_dictionary.string(sentence, bpe_symbol="subword_nmt") for sentence in sampled_indices.tolist()]
            sampled_strings = [self.tokenizer.decode(string) for string in sampled_strings]
            target_strings = [self.task.target_dictionary.string(sentence, bpe_symbol="subword_nmt") for sentence in targets.tolist()]

            # reward calculation
            with torch.no_grad():
                reward = self.compute_metric(sampled_strings, [target_strings])
            
            # masking
            if masks is not None:
                output_probs, targets = output_probs[masks], targets[masks]
                sampled_indices = sampled_indices[masks]
            
            # loss calculation
            log_probs = torch.log(output_probs)
            log_probs_of_samples = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            loss = -log_probs_of_samples * reward
            loss = loss.mean()

            # return loss
            return loss, reward
            
        
        #return mask_then_sample(outputs, targets, masks)
        return sample_then_mask(outputs, targets, masks)

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

        loss, reward = self._compute_loss(outs, tgt_tokens, masks)

        # we only want to log loss and reward to wandb during training, not during eval
        # more handling for that in reduce_metrics()
        if model.training == True:
            detached_loss = loss.detach()
            loss_to_be_logged = loss.detach()
            reward_to_be_logged = reward
        else:
            detached_loss = loss.detach()
            loss_to_be_logged = None
            reward_to_be_logged = None

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": detached_loss,
            "loss_to_be_logged": loss_to_be_logged,
            "nll_loss": detached_loss,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "reward": reward_to_be_logged
        }

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_items = [log.get("loss_to_be_logged", 0) for log in logging_outputs]
        loss_items_excl_None = [item for item in loss_items if item is not None]
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        reward_items = [log.get("reward", 0) for log in logging_outputs]
        reward_items_excl_None = [item for item in reward_items if item is not None]

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        # we only want to log loss and reward to wandb during training, not during eval
        # more handling for that in forward()
        if len(loss_items_excl_None) > 0:
            wandb_loss = sum(loss_items_excl_None) / len(loss_items_excl_None) / math.log(2)
            wandb_reward = sum(reward_items_excl_None) / len(reward_items_excl_None)
            wandb.log({"loss": wandb_loss, "reward": wandb_reward})

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