from typing import Any

import torch

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from openood.utils.config import Config



from .base_postprocessor import BasePostprocessor



class ROSSPostprocessor(BasePostprocessor):
    """
    OOD postprocessor using median, mad, cov, and ROSS score
    under input noise.
    """

    def __init__(self, config):
        super().__init__(config)
        self.args = config.postprocessor.postprocessor_args
        # number of noisy samples to generate
        self.num_samples = self.args.num_samples
        # standard deviation (magnitude) of Gaussian noise
        self.noise_magnitude = self.args.noise_magnitude
        # lambda for the gating function
        self.lambda_ = self.args.lambda_

        # --- Base Postprocessor Instantiation ---
        from openood.postprocessors.utils import get_postprocessor
        self.score_pp_name = self.args.score_postprocessor
        base_pp_config_path = f'configs/postprocessors/{self.score_pp_name}.yml'
        base_pp_config = Config(base_pp_config_path)
        base_pp_config.dataset = config.dataset
        self.base_pp = get_postprocessor(base_pp_config)

        # set APS mode from config and parse APS_score for selecting metric
        self.APS_mode = config.postprocessor.APS_mode
        self.APS_score = getattr(self.args, 'APS_score', None)
        if self.APS_mode and self.APS_score is None:
            raise ValueError("APS_mode is True but APS_score not provided for RossPostprocessor")
        # placeholder for APS score index; will resolve after metric_labels
        self.aps_score_idx = None
        # combine hyperparameter sweeps from ROSS config and base postprocessor
        var_sweep = getattr(config.postprocessor, 'postprocessor_sweep', {}) or {}
        base_sweep = getattr(self.base_pp, 'args_dict', {}) or {}
        self.args_dict = {}
        self.args_dict.update(var_sweep)
        self.args_dict.update(base_sweep)

        # replace static metric_labels with dynamic doubling
        self.metric_labels = [
            'median','mad',
            'cov', 'ross',                              
        ]

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # forward setup to the selected score postprocessor, if supported
        try:
            self.base_pp.setup(net, id_loader_dict, ood_loader_dict)
            print("Base postprocessor setup complete")
        except AttributeError:
            print("Base postprocessor setup failed")
            pass
        # Compute center and reference disagreement score (d_ref) from ID validation set
        val_loader = id_loader_dict.get('val')
        if val_loader is None:
            raise ValueError("ID validation loader 'val' not found in id_loader_dict")
        score_medians_list = []
        eps = 1e-12
        net.eval()
        with torch.no_grad():
            for batch in val_loader:
                data = batch['data'].cuda()
                # Vectorized computation of scores for noisy samples
                batch_size = data.shape[0]
                # create replicated data for all noise samples
                data_rep = data.unsqueeze(0).repeat(self.num_samples, *([1] * data.ndim))
                # generate noise and add to inputs
                noise = torch.randn_like(data_rep) * self.noise_magnitude
                noisy_data_all = data_rep + noise
                # flatten to (num_samples*batch_size, ...)
                flat_noisy = noisy_data_all.view(-1, *data.shape[1:])
                # compute scores in one batch
                _, flat_scores = self.base_pp.postprocess.__wrapped__(self.base_pp, net, flat_noisy)
                # reshape back to (num_samples, batch_size)
                score_stack = flat_scores.view(self.num_samples, batch_size)
                # compute median and MAD-based disagreement
                score_median_batch = score_stack.median(dim=0).values
                # collect statistics
                score_medians_list.extend(score_median_batch.cpu().tolist())

        # set hyperparameters based on ID validation statistics
        self.score_95_median = float(np.quantile(np.array(score_medians_list), 0.05))

        # If there are no hyperparameters to search, skip APS search
        if not self.args_dict:
            print("No hyperparameters to search, skipping APS search")
            self.hyperparam_search_done = True
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        batch_size = data.shape[0]
        data_rep = data.unsqueeze(0).repeat(self.num_samples, *([1] * data.ndim))
        noise = torch.randn_like(data_rep) * self.noise_magnitude
        flat_preds, flat_scores = self.base_pp.postprocess.__wrapped__(
            self.base_pp, net,
            (data_rep + noise).view(-1, *data.shape[1:])
        )
        score_stack = flat_scores.view(self.num_samples, batch_size).to(data.device)
        preds = flat_preds[:batch_size]
        eps = 1e-12
        score_median = score_stack.median(dim=0).values
        score_mad = torch.mean(torch.abs(score_stack - score_median), dim=0)
        score_cov = score_mad / (score_median + eps)
        min_part_mad = torch.clamp(score_median, max=self.score_95_median)
        pos_diff_mad = torch.clamp(score_median - self.score_95_median, min=0.0)
        stab_alt_mad = 1 / (score_mad + eps)
        score_min_max_95_alt_6_mad = min_part_mad + pos_diff_mad * (1 + self.lambda_ * stab_alt_mad)
        return preds, [score_median, -score_mad, -score_cov, score_min_max_95_alt_6_mad]

    def set_hyperparam(self, hyperparam: list):
        param_names = list(self.args_dict.keys())
        base_param_keys = list((getattr(self.base_pp, 'args_dict', {}) or {}).keys())
        # Assign ross-specific hyperparameters
        for idx, name in enumerate(param_names):
            if name not in base_param_keys:
                val = hyperparam[idx]
                if name == 'num_samples_list':
                    self.num_samples = int(val)
                elif name == 'noise_magnitude_list':
                    self.noise_magnitude = float(val)
                elif name == 'lambda_list':
                    self.lambda_ = float(val)
        # Delegate base postprocessor hyperparameters
        base_vals = [hyperparam[param_names.index(name)] for name in base_param_keys if name in param_names]
        if base_vals:
            self.base_pp.set_hyperparam(base_vals)

    def get_hyperparam(self):
        hyperparams = []
        base_param_keys = list((getattr(self.base_pp, 'args_dict', {}) or {}).keys())
        # Safely retrieve raw base hyperparams
        raw_base = self.base_pp.get_hyperparam() if hasattr(self.base_pp, 'get_hyperparam') else []
        # Normalize to list
        if raw_base is None:
            base_vals = []
        elif isinstance(raw_base, (list, tuple)):
            base_vals = list(raw_base)
        else:
            base_vals = [raw_base]
        base_params_dict = dict(zip(base_param_keys, base_vals))
        # Combine in args_dict order
        for name in self.args_dict.keys():
            if name == 'num_samples_list':
                hyperparams.append(self.num_samples)
            elif name == 'noise_magnitude_list':
                hyperparams.append(self.noise_magnitude)
            elif name == 'lambda_list':
                hyperparams.append(self.lambda_)
            elif name in base_params_dict:
                hyperparams.append(base_params_dict[name])
        return hyperparams

    def inference(self, net, data_loader, progress=True):
        # Perform inference; during APS search return only the chosen APS score, otherwise return all metrics
        pred_arr, conf_arr, label_arr = super().inference(net, data_loader, progress)
        if self.APS_mode and not getattr(self, 'hyperparam_search_done', False):
            # compute aps_score_idx if not already resolved
            if self.aps_score_idx is None:
                if isinstance(self.APS_score, str):
                    self.aps_score_idx = self.metric_labels.index(self.APS_score)
                else:
                    self.aps_score_idx = int(self.APS_score)
            # select only the APS_score metric
            if isinstance(conf_arr, np.ndarray) and conf_arr.ndim == 2:
                conf_arr = conf_arr[:, self.aps_score_idx]
        # otherwise, return all metrics as-is
        return pred_arr, conf_arr, label_arr