# roberta-crf

[English](https://github.com/cedar33/roberta-crf/blob/master/README.md)
[简体中文]([English](https://github.com/cedar33/roberta-crf/blob/master/README.md)

a graceful method to apply crf into reberta model for sentence labeling task

The main purpose of this repository is trying to modify the roberta's source code as less as possible to apply crf into roberta model.Before running this code, you are supposed to clone [fairseq](https://github.com/pytorch/fairseq).

1. clone [fairseq](https://github.com/pytorch/fairseq).
2. replace `fairseq/fairseq/models/roberta/model.py`, `fairseq/fairseq/models/roberta/hub_interface.py`, `fairseq/fairseq/trainer.py`, do not worry about the changes may influence other tasks, the differences will be shown below.
3. move files in folder tasks and criterions into `fairseq/fairseq/tasks` and `fairseq/fairseq/criterions`
4. run `kaggel_ner_encoder.py` to encode ner data
5. run `pre_process.sh` to make traning and vilidating file
6. run `train.sh` to finetuning roberta

click [here](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus) to download kaggle ner data

difference in `trainer.py`
```diff
            try:
                self.get_model().load_state_dict(
                    state["model"], strict=True, args=self.args
                )
                if utils.has_parameters(self.get_criterion()):
+                    state["criterion"] = self.get_criterion().state_dict()  # add this code to load crf layer while loading state dict
                    self.get_criterion().load_state_dict(
                        state["criterion"], strict=True
                    )
```

difference in `hub_interface.py`
```diff
+    def predict_label(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
+        features = self.extract_features(tokens.to(device=self.device))
+        path_score, path = self.model.labeling_heads[head].forward_decode(features)
+        return path_score, path
```

difference in `model.py`
```diff
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
+from fairseq.modules.dynamic_crf_layer import DynamicCRF as CRF

from .hub_interface import RobertaHubInterface


logger = logging.getLogger(__name__)


@register_model('roberta')
class RobertaModel(FairseqEncoderModel):
  ...

        self.classification_heads = nn.ModuleDict()
+        self.labeling_heads = nn.ModuleDict()

    @staticmethod
-    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, classification_head_name=None, **kwargs):
-        if classification_head_name is not None:
+    # add arg `labeling_head_name`
+    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, classification_head_name=None, labeling_head_name=None, **kwargs):
+        if (classification_head_name or labeling_head_name) is not None:
            features_only = True

        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
+        if labeling_head_name is not None:
+            x = self.labeling_heads[labeling_head_name](x, **kwargs)
        return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)
    ...
+    # according to `register_classificaation_head`
+    def register_labeling_head(self, name, num_tags=None, inner_dim=None, **kwargs):
+        """Register a labeling head."""
+        if name in self.labeling_heads:
+            prev_num_tags = self.labeling_heads[name].dense.out_features
+            prev_inner_dim = self.labeling_heads[name].dense.in_features
+            if num_tags != prev_num_tags or inner_dim != prev_inner_dim:
+                logger.warning(
+                    're-registering head "{}" with num_tags {} (prev: {}) '
+                    'and inner_dim {} (prev: {})'.format(
+                        name, num_tags, prev_num_tags, inner_dim, prev_inner_dim
+                    )
+                )
+        self.labeling_heads[name] = RobertaLabelingHead(
+            self.args.encoder_embed_dim,
+            inner_dim or self.args.encoder_embed_dim,
+            num_tags,
+            self.args.pooler_dropout,
+            self.args.quant_noise_pq,
+            self.args.quant_noise_pq_block_size,
+        )
    ...
    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + 'decoder'):
                new_k = prefix + 'encoder' + k[len(prefix + 'decoder'):]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
-        current_head_names = (
-            [] if not hasattr(self, 'classification_heads')
-            else self.classification_heads.keys()
-        )
+        if hasattr(self, 'classification_heads'):
+            current_head_names = (self.classification_heads.keys())
+        elif hasattr(self, 'labeling_heads'):
+            current_head_names = (self.labeling_heads.keys())
+        else:
+            current_head_names = ([])
        keys_to_delete = []
        for k in state_dict.keys():
-             if not k.startswith(prefix + 'classification_heads.'):
-                continue
-
-            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
-            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
-            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)
+            if not (k.startswith(prefix + 'classification_heads.') or k.startswith(prefix + 'labeling_heads.')):
+                continue
+            elif k.startswith(prefix + 'classification_heads.'):
+                head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
+                num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
+                inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)
+            elif k.startswith(prefix + 'labeling_heads.'):
+                head_name = k[len(prefix + 'labeling_heads.'):].split('.')[0]
+                num_classes = state_dict[prefix + 'labeling_heads.' + head_name + '.dense.weight'].size(0)
+                inner_dim = state_dict[prefix + 'labeling_heads.' + head_name + '.dense.weight'].size(1)

            if getattr(self.args, 'load_checkpoint_heads', False):
-                if head_name not in current_head_names:
-                    self.register_classification_head(head_name, num_classes, inner_dim)
+                if (head_name not in current_head_names 
+                    and k.startswith(prefix + 'classification_heads.')):
+                    self.register_classification_head(head_name, num_classes, inner_dim)
+                elif (head_name not in current_head_names 
+                      and k.startswith(prefix + 'labeling_heads.')):
+                    self.register_labeling_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ) 
+                or (num_classes != self.labeling_heads[head_name].dense.weight):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v
+        if hasattr(self, 'labeling_heads'):
+            cur_state = self.labeling_heads.state_dict()
+            for k, v in cur_state.items():
+                if prefix + 'labeling_heads.' + k not in state_dict:
+                    state_dict[prefix + 'labeling_heads.' + k] = v
    ...
```
