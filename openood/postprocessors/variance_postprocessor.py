from typing import Any

import torch

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from openood.utils.config import Config



from .base_postprocessor import BasePostprocessor



class VariancePostprocessor(BasePostprocessor):
    """
    OOD postprocessor using mean, median, variance, entropy, and confidence multipliers of scores
    under input noise, now including logit-based disagreement scores.
    """

    def __init__(self, config):
        super().__init__(config)
        self.args = config.postprocessor.postprocessor_args
        # number of noisy samples to generate
        self.num_samples = self.args.num_samples
        # standard deviation (magnitude) of Gaussian noise
        self.noise_magnitude = self.args.noise_magnitude

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
            raise ValueError("APS_mode is True but APS_score not provided for VariancePostprocessor")
        # placeholder for APS score index; will resolve after metric_labels
        self.aps_score_idx = None
        # combine hyperparameter sweeps from variance config and base postprocessor
        var_sweep = getattr(config.postprocessor, 'postprocessor_sweep', {}) or {}
        base_sweep = getattr(self.base_pp, 'args_dict', {}) or {}
        self.args_dict = {}
        self.args_dict.update(var_sweep)
        self.args_dict.update(base_sweep)

        # attach metric labels
        # Initialize gating (w) and adjustment (lambda_) hyperparameters
        self.w = 0.5
        self.lambda_ = 5.0

        # placeholders for median and MAD-based hyperparameters
        self.score_95_median = None
        self.d_ref_mad = None
        self.w_median = 0.5
        self.lambda_mad = 5.0

        # replace static metric_labels with dynamic doubling
        self.metric_labels = [
            'score_mean', 'score_median',
            'score_base_std_deviation', 'score_base_mad',
            'score_base_coefficient_of_variation', 'score_base_mad_cov',    
            'score_base_coefficient_of_variation_ratio', 'score_base_mad_cov_ratio',
            'score_base_interquartile_range', 'score_base_median_interquartile_range',
            'score_final', 'score_final_median', 
            'score_min_max_95', 'score_min_max_95_median',
            'score_min_max_95_alt_1', 'score_min_max_95_alt_1_median',
            'score_min_max_95_alt_2', 'score_min_max_95_alt_2_median',  
            'score_min_max_95_alt_3', 'score_min_max_95_alt_3_median',  
            'score_min_max_95_alt_4', 'score_min_max_95_alt_4_median',          
            'score_min_max_95_alt_5', 'score_min_max_95_alt_5_median',                          
            'score_min_max_95_alt_6', 'score_min_max_95_alt_6_median',                              
            'score_min_max_95_alt_7', 'score_min_max_95_alt_7_median',
            'score_min_max_95_alt_8', 'score_min_max_95_alt_8_median',
            'score_min_max_sigmoid', 'score_min_max_sigmoid_median',
            'score_harmonic', 'score_harmonic_median',
            'score_harmonic_basic', 'score_harmonic_basic_median',  
        ]
        # Initialize gating (w) and adjustment (lambda_) hyperparameters
        # placeholders for center and reference disagreement score (to be computed in setup)
        self.score_centre = None
        self.score_95 = None
        self.score_99 = None
        self.d_ref = None

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
        score_means = []
        d_vals = []
        score_medians_list = []
        d_vals_mad = []
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
                score_mean = score_stack.mean(dim=0)
                score_std = torch.sqrt(score_stack.var(dim=0, unbiased=False))
                score_cov = score_std / (score_mean + eps)
                # compute median and MAD-based disagreement
                score_median_batch = score_stack.median(dim=0).values
                score_mad_batch = torch.mean(torch.abs(score_stack - score_median_batch), dim=0)
                med_mad_cov = score_mad_batch / (score_median_batch + eps)
                # collect statistics
                score_means.extend(score_mean.cpu().tolist())
                d_vals.extend(score_cov.cpu().tolist())
                score_medians_list.extend(score_median_batch.cpu().tolist())
                d_vals_mad.extend(med_mad_cov.cpu().tolist())

        # set hyperparameters based on ID validation statistics
        self.score_95 = float(np.quantile(np.array(score_means), 0.05))
        self.d_ref = float(np.quantile(np.array(d_vals), 0.05))
        self.score_95_median = float(np.quantile(np.array(score_medians_list), 0.05))
        self.d_ref_mad = float(np.quantile(np.array(d_vals_mad), 0.05))

        # Compute gating width (w) and median gating width for MAD branch
        score_means_arr = np.array(score_means)
        score_medians_arr = np.array(score_medians_list)
        self.w = float(np.std(score_means_arr))
        self.w_median = float(np.std(score_medians_arr))
        # adjustment scales
        self.lambda_ = 0.1
        self.lambda_mad = 0.1

        # If there are no hyperparameters to search, skip APS search
        if not self.args_dict:
            print("No hyperparameters to search, skipping APS search")
            self.hyperparam_search_done = True
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # Vectorized computation of scores and predictions for noisy samples
        batch_size = data.shape[0]
        # replicate inputs across noise samples
        data_rep = data.unsqueeze(0).repeat(self.num_samples, *([1] * data.ndim))
        # generate noise and add
        noise = torch.randn_like(data_rep) * self.noise_magnitude
        noisy_data_all = data_rep + noise
        # flatten to process in one batch
        flat_noisy = noisy_data_all.view(-1, *data.shape[1:])
        # forward through base postprocessor
        flat_preds, flat_scores = self.base_pp.postprocess.__wrapped__(self.base_pp, net, flat_noisy)
        # reshape scores to (num_samples, batch_size)
        score_stack = flat_scores.view(self.num_samples, batch_size).to(data.device)
        # extract original predictions
        preds = flat_preds[:batch_size]
        eps = 1e-12

        # 1. Score mean
        score_mean = score_stack.mean(dim=0)
        # 2. Score median
        score_median = score_stack.median(dim=0).values
        # 3. Score standard deviation
        score_variance = score_stack.var(dim=0, unbiased=False)
        score_std_deviation = torch.sqrt(score_variance)
        # 4. Score coefficient of variation
        score_coefficient_of_variation = score_std_deviation / (score_mean + eps)
        # compute MAD-based variation (mean absolute deviation from median)
        score_mad = torch.mean(torch.abs(score_stack - score_median), dim=0)
        score_mad_cov = score_mad / (score_median + eps)
        # insert base MAD after std deviation
        score_base_mad = score_mad
        #score radius
        score_radius = torch.max(score_stack, dim=0).values - torch.min(score_stack, dim=0).values

        # Compute final score with sigmoid gating and stability adjustment (mean/std)
        sigma_input = (score_mean - self.score_95) / self.w
        gate = torch.sigmoid(sigma_input)
        stability_adj = self.lambda_ * (1.0 - score_coefficient_of_variation / (self.d_ref + eps))
        score_final = score_mean + gate * stability_adj
        # Compute median/MAD-based final score
        sigma_input_med = (score_median - self.score_95_median) / self.w_median
        gate_med = torch.sigmoid(sigma_input_med)
        stability_adj_med = self.lambda_mad * (1.0 - score_mad_cov / (self.d_ref_mad + eps))
        score_final_mad = score_median + gate_med * stability_adj_med

        # Compute min-max blended score: min(S_base, center) + max(0, S_base - center)*(1/C)
        stab = 1/(score_coefficient_of_variation+eps)
        min_part = torch.clamp(score_mean, max=self.score_95)
        pos_diff = torch.clamp(score_mean - self.score_95, min=0.0)
        score_min_max_95 = score_mean + pos_diff * 0.01 * stab
        # Alternative min-max blended score
        stab_alt = 1/(score_std_deviation+eps)
        score_min_max_95_alt_1 = score_mean + pos_diff * (1+0.001 * stab_alt)
        # Alternative min-max blended score
        stab_alt = 1/(score_std_deviation+eps)
        score_min_max_95_alt_2 = score_mean + pos_diff * (1+0.003 * stab_alt)
        # Alternative min-max blended score
        stab_alt = 1/(score_std_deviation+eps)
        score_min_max_95_alt_3 = score_mean + pos_diff * (1+0.005 * stab_alt)
        score_min_max_95_alt_4 = min_part + pos_diff * (1+0.005 * stab_alt)
        # Alternative min-max blended score
        stab_alt = 1/(score_std_deviation+eps)
        score_min_max_95_alt_5 = score_mean + pos_diff * (1+0.01 * stab_alt)
        # Alternative min-max blended score
        stab_alt = 1/(score_std_deviation+eps)
        score_min_max_95_alt_6 = min_part + pos_diff * (1+0.01 * stab_alt)
        # Alternative min-max blended score
        stab_alt = 1/(score_std_deviation+eps)
        score_min_max_95_alt_7 = min_part + pos_diff * (1+0.02 * stab_alt)
        # Alternative min-max blended score
        stab_alt = 1/(score_std_deviation+eps)
        score_min_max_95_alt_8 = min_part + pos_diff * (1+0.05 * stab_alt)

        # Sigmoid-gated min-max blended score
        g_sig = torch.sigmoid((score_mean - self.score_95)/(self.w + eps))
        score_min_max_sigmoid = min_part + pos_diff * g_sig * self.lambda_ * stab
        # Harmonic mean score: combine confidence and stability
        stability_ratio = self.lambda_ * score_mean/(score_coefficient_of_variation+eps)
        score_harmonic = 2*(score_mean * stability_ratio)/(score_mean + stability_ratio + eps)
        # Harmonic mean score: combine confidence and stability
        stability = 1/(score_std_deviation+eps)
        score_harmonic_basic = (score_mean * stability)/(score_mean + stability + eps)
        


        # Interquartile Range (IQR)
        q75 = torch.quantile(score_stack, 0.75, dim=0)
        q25 = torch.quantile(score_stack, 0.25, dim=0)
        score_interquartile_range = q75 - q25

        # Alternative weighted geometric median score (base^(-base) * stability^base)
        # C = 1 / score_coefficient_of_variation
        # # We work in log-space to prevent explosion
        # log_C_stable = torch.log(C + eps)
        # log_score_mean = torch.log(score_mean + eps)

        # score_final_alt_1 = log_score_mean * log_C_stable
        # Median/MAD variant of weighted geometric median score
        # C_mad = 1 / score_mad_cov
        # log_C_stable_mad = torch.log(C_mad + eps)
        # log_score_median = torch.log(score_median + eps)
        # score_final_alt_1_mad = log_score_median * log_C_stable_mad
        # Prepare median/MAD composite variables
        min_part_mad = torch.clamp(score_median, max=self.score_95_median)
        pos_diff_mad = torch.clamp(score_median - self.score_95_median, min=0.0)
        stab_mad = 1/(score_mad_cov + eps)
        stab_alt_mad = 1/(score_mad + eps)
        g_sig_med = torch.sigmoid((score_median - self.score_95_median)/(self.w_median + eps))
        # Median/MAD min-max blended scores
        score_min_max_95_mad = score_median + pos_diff_mad * 0.01 * stab_mad
        score_min_max_95_alt_1_mad = score_median + pos_diff_mad * (1+0.001 * stab_alt_mad)
        score_min_max_95_alt_2_mad = score_median + pos_diff_mad * (1+0.003 * stab_alt_mad)
        score_min_max_95_alt_3_mad = score_median + pos_diff_mad * (1+0.005 * stab_alt_mad)
        score_min_max_95_alt_4_mad = min_part_mad + pos_diff_mad * (1+0.005 * stab_alt_mad)
        score_min_max_95_alt_5_mad = score_median + pos_diff_mad * (1+0.01 * stab_alt_mad)
        score_min_max_95_alt_6_mad = min_part_mad + pos_diff_mad * (1+0.01 * stab_alt_mad)
        score_min_max_95_alt_7_mad = min_part_mad + pos_diff_mad * (1+0.02 * stab_alt_mad)
        score_min_max_95_alt_8_mad = min_part_mad + pos_diff_mad * (1+0.05 * stab_alt_mad)
        # Median/MAD sigmoid-gated min-max blended score
        score_min_max_sigmoid_mad = min_part_mad + pos_diff_mad * g_sig_med * self.lambda_mad * stab_mad
        # Median/MAD harmonic mean scores
        stability_ratio_mad = self.lambda_mad * score_median/(score_mad_cov + eps)
        score_harmonic_mad = 2*(score_median * stability_ratio_mad)/(score_median + stability_ratio_mad + eps)
        stability_mad_basic = 1/(score_mad + eps)
        score_harmonic_basic_mad = (score_median * stability_mad_basic)/(score_median + stability_mad_basic + eps)
        return preds, [
            score_mean, score_median,
            -score_std_deviation, -score_mad,
            -score_coefficient_of_variation, -score_mad_cov,
            score_mean/(score_coefficient_of_variation+eps), score_median/(score_mad_cov+eps),
            score_mean-score_interquartile_range, score_median-score_mad,
            score_final, score_final_mad,
            score_min_max_95, score_min_max_95_mad,
            score_min_max_95_alt_1, score_min_max_95_alt_1_mad,
            score_min_max_95_alt_2, score_min_max_95_alt_2_mad,
            score_min_max_95_alt_3, score_min_max_95_alt_3_mad,
            score_min_max_95_alt_4, score_min_max_95_alt_4_mad,
            score_min_max_95_alt_5, score_min_max_95_alt_5_mad,
            score_min_max_95_alt_6, score_min_max_95_alt_6_mad,
            score_min_max_95_alt_7, score_min_max_95_alt_7_mad,
            score_min_max_95_alt_8, score_min_max_95_alt_8_mad,
            score_min_max_sigmoid, score_min_max_sigmoid_mad,
            score_harmonic, score_harmonic_mad,
            score_harmonic_basic, score_harmonic_basic_mad,
        ]

    def set_hyperparam(self, hyperparam: list):
        param_names = list(self.args_dict.keys())
        base_param_keys = list((getattr(self.base_pp, 'args_dict', {}) or {}).keys())
        # Assign variance-specific hyperparameters
        for idx, name in enumerate(param_names):
            if name not in base_param_keys:
                val = hyperparam[idx]
                if name == 'num_samples_list':
                    self.num_samples = int(val)
                elif name == 'noise_magnitude_list':
                    self.noise_magnitude = float(val)
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