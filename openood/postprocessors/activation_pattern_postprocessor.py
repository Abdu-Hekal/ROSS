from typing import Any, Dict, Set
import torch
import torch.nn as nn
from .base_postprocessor import BasePostprocessor

class ActivationPatternPostprocessor(BasePostprocessor):
    """
    Postprocessor implementing Activation Pattern Matching OOD detection.
    """
    def __init__(self, config):
        super().__init__(config)
        args = config.postprocessor.postprocessor_args
        self.layer_name = getattr(args, 'layer_name', None)
        if self.layer_name is None:
            raise ValueError("layer_name must be specified in postprocessor_args for ActivationPatternPostprocessor")
        self.threshold = getattr(args, 'threshold', 0.0)
        self.freq = getattr(args, 'freq', 0.5)
        self.use_return_feature = (self.layer_name == 'features')
        self.module = None
        self.class_profiles: Dict[int, Set[int]] = {}

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.use_return_feature:
            activations = []
            labels_list = []
            val_loader = id_loader_dict.get('val')
            if val_loader is None:
                raise ValueError("ID validation loader 'val' not found")
            net.eval()
            with torch.no_grad():
                for batch in val_loader:
                    data = batch['data'].cuda()
                    labels = batch['label'].cpu()
                    labels_list.append(labels)
                    _, feats = net(data, return_feature=True)
                    activations.append(feats.detach().cpu())
            all_feats = torch.cat(activations, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            for c in torch.unique(all_labels).tolist():
                mask = all_labels == c
                feats_c = all_feats[mask]
                if feats_c.numel() == 0:
                    self.class_profiles[int(c)] = set()
                else:
                    freq_vals = (feats_c > self.threshold).float().mean(dim=0)
                    active_idxs = torch.nonzero(freq_vals >= self.freq, as_tuple=True)[0].tolist()
                    self.class_profiles[int(c)] = set(active_idxs)
            self.metric_labels = ['activation_pattern_jaccard']
            return
        # find the specified layer module
        for name, module in net.named_modules():
            if name == self.layer_name:
                self.module = module
                break
        if self.module is None:
            raise ValueError(f"Layer {self.layer_name} not found in model")
        # collect activations and labels from ID validation set
        activations = []
        labels_list = []
        def hook_fn(module, inp, output):
            f = output.detach().flatten(start_dim=1).cpu()
            activations.append(f)
        handle = self.module.register_forward_hook(hook_fn)
        val_loader = id_loader_dict.get('val')
        if val_loader is None:
            raise ValueError("ID validation loader 'val' not found")
        net.eval()
        with torch.no_grad():
            for batch in val_loader:
                data = batch['data'].cuda()
                labels = batch['label'].cpu()
                labels_list.append(labels)
                _ = net(data)
        handle.remove()
        # build class profiles based on activation frequency
        all_feats = torch.cat(activations, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        for c in torch.unique(all_labels).tolist():
            mask = all_labels == c
            feats_c = all_feats[mask]
            if feats_c.numel() == 0:
                self.class_profiles[int(c)] = set()
            else:
                freq_vals = (feats_c > self.threshold).float().mean(dim=0)
                active_idxs = torch.nonzero(freq_vals >= self.freq, as_tuple=True)[0].tolist()
                self.class_profiles[int(c)] = set(active_idxs)
        # set metric label
        self.metric_labels = ['activation_pattern_jaccard']

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        if self.use_return_feature:
            output, feats = net(data, return_feature=True)
            feats = feats.detach().cpu()
            batch_size = feats.shape[0]
            jaccard_scores = []
            for i in range(batch_size):
                s = set(torch.nonzero(feats[i] > self.threshold, as_tuple=True)[0].tolist())
                best_sim = 0.0
                for cp in self.class_profiles.values():
                    inter = len(s & cp)
                    union = len(s | cp)
                    sim = inter / union if union > 0 else 0.0
                    if sim > best_sim:
                        best_sim = sim
                jaccard_scores.append(best_sim)
            jaccard_tensor = torch.tensor(jaccard_scores)
            preds = output.argmax(dim=1)
            return preds, [jaccard_tensor]
        # capture activations for the current batch
        current_acts = []
        def hook_fn(module, inp, output):
            current_acts.append(output.detach().flatten(start_dim=1).cpu())
        handle = self.module.register_forward_hook(hook_fn)
        output = net(data)
        handle.remove()
        feats = current_acts[0]
        # compute Jaccard similarity between sample and each class profile
        batch_size = feats.shape[0]
        jaccard_scores = []
        for i in range(batch_size):
            s = set(torch.nonzero(feats[i] > self.threshold, as_tuple=True)[0].tolist())
            best_sim = 0.0
            for cp in self.class_profiles.values():
                inter = len(s & cp)
                union = len(s | cp)
                sim = inter / union if union > 0 else 0.0
                if sim > best_sim:
                    best_sim = sim
            jaccard_scores.append(best_sim)
        jaccard_tensor = torch.tensor(jaccard_scores)
        preds = output.argmax(dim=1)
        return preds, [jaccard_tensor] 