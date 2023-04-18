
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions.nat_loss import LabelSmoothedDualImitationCriterion

from dataclasses import dataclass, field

@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="bleu",
                                       metadata={"help": "sentence level metric"})


@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(LabelSmoothedDualImitationCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task, label_smoothing=0.0)
        self.metric = sentence_level_metric

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """

        def compute_bleu(hypothesis, reference):
          from sacrebleu.metrics import BLEU
          bleu = BLEU(smooth_method="add-k", smooth_value="1")
          bleu.corpus_score(hypothesis, reference)
          return bleu.score
          
        def compute_chrf(hypothesis, reference):
          # maybe factorize code on a higher level
          pass

        logging_output = ""

        #padding mask, do not remove
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        #we take a softmax over outputs
        outputs_softmax = F.log_softmax(outputs, dim=1)
        
        #argmax over the softmax \ sampling (e.g. multinomial)
        #sampled_sentence = [4, 17, 18, 19, 20]
        sampled_sentence = torch.multinomial(outputs_softmax) # isn't multinomial sampling instead of argmax?

        #sampled_sentence_string = tgt_dict.string([4, 17, 18, 19, 20])
        #see dictionary class of fairseq
        sampled_sentence_string = self.tgt_dict.string()
        
        sampled_sentence_string = sampled_sentence.string(sampled_sentence)

        #target_sentence = "I am a sentence"
        #with torch.no_grad()
            #R(*) = eval_metric(sampled_sentence_string, target_sentence)
            #R(*) is a number, BLEU, сhrf, etc.

        # do we calculate the loss per sentence or per batch?
        with torch.no_grad():
          if self.metric == "BLEU":
            R = compute_bleu(sampled_sentence_string, target_sentence)
          elif self.metric == "ChrF":
            R = compute_chrf(sampled_sentence_string, target_sentence)

        #loss = -log_prob(outputs)*R()
        #loss = loss.mean()
        loss = -log_prob(outputs) * R
        loss = loss.mean()

        sample_size = 1 # what do we return here?

        return loss, sample_size, logging_output