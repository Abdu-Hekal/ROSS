from typing import Any
import torch
import torch.nn as nn
from copy import deepcopy
import os
import numpy as np

from openood.postprocessors import BasePostprocessor

class MinMaxPostprocessor(BasePostprocessor):
    """
    OOD postprocessor using min-max blending of scores under input noise,
    including logit-based disagreement scores and other uncertainty metrics.
    """
    def __init__(self, config):
        super().__init__(config)
        self.args = config.postprocessor.postprocessor_args
        # number of noisy samples to generate
        self.num_samples = self.args.num_samples
        # standard deviation (magnitude) of Gaussian noise
        self.noise_magnitude = self.args.noise_magnitude
        # score postprocessor name
        self.score_pp_name = self.args.score_postprocessor
        # instantiate the base postprocessor exactly as if called directly, preserving hyperparameters
        from openood.postprocessors import get_postprocessor as _get_pp
        base_config = deepcopy(config)
        base_config.postprocessor.name = self.score_pp_name
        self.base_pp = _get_pp(base_config)
        
        # attach metric labels
        self.metric_labels = [
            # Original 10 metrics
            'score_mean',
            'score_median',
            'score_base_variance',
            'score_base_std_deviation',
            'score_base_mean_absolute_deviation',
            'score_base_coefficient_of_variation',
            'score_base_coefficient_of_variation_squared',  
            'score_base_squared_coefficient_of_variation',
            'score_base_avg_pairwise_distance',
            'score_base_mutual_information',
            'score_base_mean_js_divergence',
            # New 3 metrics from the paper
            'score_base_disagreement_score',
            'score_base_weight_entropy',
            'score_base_std_log_logits',
            'score_final',
            'score_min_max',
            'score_localized',
            'score_hybrid',
            'score_flip',
        ]
        # Initialize gating (w) and adjustment (lambda_) hyperparameters
        self.w = 0.5
        self.lambda_ = 5.0
        # placeholders for center and reference disagreement score (to be computed in setup)
        self.center = None
        self.d_ref = None
        # number of neighbors for local reference (default to 10 if not specified)
        # ensure k is int and nonzero, defaulting to 10
        self.k = int(getattr(self.args, 'k', None) or 10)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # forward setup to the selected score postprocessor, if supported
        try:
            self.base_pp.setup(net, id_loader_dict, ood_loader_dict)
        except AttributeError:
            pass
        # Compute center and reference disagreement score (d_ref) from ID validation set
        val_loader = id_loader_dict.get('val')
        if val_loader is None:
            raise ValueError("ID validation loader 'val' not found in id_loader_dict")
        score_means = []
        d_vals = []
        eps = 1e-12
        net.eval()
        with torch.no_grad():
            for batch in val_loader:
                data = batch['data'].cuda()
                # base and noisy scores
                _, base_score = self.base_pp.postprocess(net, data)
                scores = [base_score]
                for _ in range(self.num_samples - 1):
                    noisy_data = data + torch.normal(mean=torch.zeros_like(data), std=self.noise_magnitude)
                    _, score_i = self.base_pp.postprocess(net, noisy_data)
                    scores.append(score_i)
                score_stack = torch.stack(scores, dim=0)
                score_mean = score_stack.mean(dim=0)
                score_std = torch.sqrt(score_stack.var(dim=0, unbiased=False))
                score_cov = score_std / (score_mean + eps)
                score_means.extend(score_mean.cpu().tolist())
                d_vals.extend(score_cov.cpu().tolist())
        # set hyperparameters based on ID validation statistics
        self.center = float(np.quantile(np.array(score_means), 0.05))
        self.d_ref = float(np.mean(np.array(d_vals)))
        # Compute gating width (w) and adjustment scale (lambda_) from validation set
        score_means_arr = np.array(score_means)
        d_vals_arr = np.array(d_vals)
        stab_base_arr = 1.0 - d_vals_arr / (self.d_ref + eps)
        # Set gating width to std of score means
        self.w = float(np.std(score_means_arr))
        # Set lambda as ratio of std of base scores to std of stability adjustment
        self.lambda_ = float(np.std(score_means_arr) / (np.std(stab_base_arr) + eps))
        # Build memory bank for local disagreement reference
        self.memory_features = []
        self.memory_d = []
        net.eval()
        with torch.no_grad():
            for batch in id_loader_dict.get('train', []):
                data = batch['data'].cuda()
                # feature extraction (use logits as features)
                output = net(data)
                # compute dynamic disagreement D for each sample
                scores = []
                _, base_score_batch = self.base_pp.postprocess(net, data)
                scores.append(base_score_batch)
                for _ in range(self.num_samples - 1):
                    noisy_data = data + torch.normal(mean=torch.zeros_like(data), std=self.noise_magnitude)
                    _, score_i_batch = self.base_pp.postprocess(net, noisy_data)
                    scores.append(score_i_batch)
                score_stack_batch = torch.stack(scores, dim=0)
                score_mean_batch = score_stack_batch.mean(dim=0)
                score_std_batch = torch.sqrt(score_stack_batch.var(dim=0, unbiased=False))
                D_batch = score_std_batch / (score_mean_batch + eps)
                self.memory_features.append(output.cpu())
                self.memory_d.append(D_batch.cpu())
        if len(self.memory_features) > 0:
            self.memory_features = torch.cat(self.memory_features, dim=0)
            self.memory_d = torch.cat(self.memory_d, dim=0)
        else:
            raise ValueError("ID training loader 'train' not found or empty for memory bank.")

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # get base predictions and scores from selected postprocessor
        preds, base_score = self.base_pp.postprocess(net, data)
        # collect preds and scores from noisy samples
        scores = [base_score]
        
        # collect logits and probability distributions for uncertainty metrics
        output0 = net(data)
        logit_list = [output0]
        prob_list = [torch.softmax(output0, dim=1)]
        
        for _ in range(self.num_samples - 1):
            noisy_data = data + torch.normal(mean=torch.zeros_like(data), std=self.noise_magnitude)
            _, score_i = self.base_pp.postprocess(net, noisy_data)
            scores.append(score_i)
            # collect logits and probability distributions for uncertainty metrics
            output_i = net(noisy_data)
            logit_list.append(output_i)
            prob_list.append(torch.softmax(output_i, dim=1))

        # stack scores: (num_samples, batch_size)
        score_stack = torch.stack(scores, dim=0)
        eps = 1e-12

        # 1. Score mean
        score_mean = score_stack.mean(dim=0)
        # 2. Score median
        score_median = score_stack.median(dim=0).values
        # 3. Score variance
        score_variance = score_stack.var(dim=0, unbiased=False)
        # 4. Score standard deviation
        score_std_deviation = torch.sqrt(score_variance)
        # 5. Score mean absolute deviation
        score_mean_absolute_deviation = torch.mean(torch.abs(score_stack - score_mean.unsqueeze(0)), dim=0)
        # 6. Score coefficient of variation
        score_coefficient_of_variation = score_std_deviation / (score_mean + eps)
        # 7. Score average pairwise distance
        score_permuted = score_stack.permute(1, 0).unsqueeze(-1)
        score_distances = torch.cdist(score_permuted, score_permuted, p=2)
        Ns = score_distances.shape[1]
        idx_i, idx_j = torch.triu_indices(Ns, Ns, offset=1)
        pairwise_score_dist = score_distances[:, idx_i, idx_j]
        score_avg_pairwise_distance = pairwise_score_dist.mean(dim=1)
        
        # --- Compute probability-based metrics ---
        prob_stack = torch.stack(prob_list, dim=0)
        p_mean = prob_stack.mean(dim=0)
        # 8. Mutual Information
        predictive_entropy = -(p_mean * torch.log(p_mean + eps)).sum(dim=1)
        entropies = -(prob_stack * torch.log(prob_stack + eps)).sum(dim=2)
        expected_entropy = entropies.mean(dim=0)
        mutual_information = predictive_entropy - expected_entropy
        
        # 9. Mean Jensen-Shannon Divergence
        prob_permuted = prob_stack.permute(1, 0, 2)
        Np = prob_stack.shape[0]
        idx_i, idx_j = torch.triu_indices(Np, Np, offset=1)
        p_i = prob_permuted[:, idx_i, :]
        p_j = prob_permuted[:, idx_j, :]
        m_ij = 0.5 * (p_i + p_j)
        kl1 = (p_i * (torch.log(p_i + eps) - torch.log(m_ij + eps))).sum(dim=2)
        kl2 = (p_j * (torch.log(p_j + eps) - torch.log(m_ij + eps))).sum(dim=2)
        js_div = 0.5 * (kl1 + kl2)
        mean_js_divergence = js_div.mean(dim=1)
        
        # --- Compute the 3 new logit-based disagreement scores ---
        logit_stack = torch.stack(logit_list, dim=0) # Shape: (num_samples, batch_size, num_classes)
        
        # Get the logit value corresponding to the predicted class for each sample.
        # The predicted class is the argmax of the mean predictive probability.
        predicted_class = torch.argmax(p_mean, dim=1) # Shape: (batch_size)
        # Use gather to select the correct logit for each sample in the batch
        # Unsqueeze and expand to match dimensions for gather
        predicted_class_expanded = predicted_class.view(1, -1, 1).expand(self.num_samples, -1, 1)
        max_logits = logit_stack.gather(2, predicted_class_expanded).squeeze(2) # Shape: (num_samples, batch_size)

        # Apply logit truncation as per Equation (8) in the paper
        z_star = torch.clamp(max_logits, min=eps)

        # Normalize truncated logits to get weights (eta_tilde), Equation (9)
        z_star_sum = z_star.sum(dim=0, keepdim=True)
        eta_tilde = z_star / (z_star_sum + eps)
        
        # 10. Disagreement Score (DS), Equation (9)
        disagreement_score = 1.0 / (torch.sum(eta_tilde**2, dim=0) + eps)
        
        # 11. Weight Entropy (WE), Equation (10)
        weight_entropy = -torch.sum(eta_tilde * torch.log(eta_tilde + eps), dim=0)
        
        # 12. Standard Deviation of Log-Logits (Std of LLs), Equation (11)
        std_log_logits = -torch.std(torch.log(z_star + eps), dim=0, unbiased=True)

        # Compute final score with sigmoid gating and stability adjustment
        sigma_input = (score_mean - self.center) / self.w
        gate = torch.sigmoid(sigma_input)
        stability_adj = self.lambda_ * (1.0 -  score_coefficient_of_variation / (self.d_ref + eps))
        score_final = score_mean + gate * stability_adj

        # Compute min-max blended score: min(S_base, center) + max(0, S_base - center)*(1 + lambda * S_stab)
        # S_stab = 1 - D / d_ref
        stab = 1.0 - score_coefficient_of_variation / (self.d_ref + eps)
        min_part = torch.clamp(score_mean, max=self.center)
        pos_diff = torch.clamp(score_mean - self.center, min=0.0)
        score_min_max = min_part + pos_diff * (1.0 + self.lambda_ * stab)

        # --- Idea 1: Localized Bonus/Malus ---
        mem_feats = self.memory_features.to(data.device)
        mem_d = self.memory_d.to(data.device)
        # normalize features for cosine similarity
        feat_norm = output0 / (output0.norm(dim=1, keepdim=True) + eps)
        mem_norm = mem_feats / (mem_feats.norm(dim=1, keepdim=True) + eps)
        sims = torch.matmul(feat_norm, mem_norm.T)
        _, idx = sims.topk(self.k, dim=1)
        d_nn = mem_d[idx]
        d_ref_local = d_nn.mean(dim=1)
        stability_adj_local = self.lambda_ * (1.0 - d_ref_local / (score_coefficient_of_variation + eps))
        score_localized = score_mean + gate * stability_adj_local

        # --- Idea 2: Hybrid Guidance Fusion ---
        C = torch.exp(-score_coefficient_of_variation / (self.d_ref + eps))
        G = sims.mean(dim=1)
        score_hybrid = score_mean * C * G

        # --- Idea 3: Disagreement-Guided Confidence ---
        D_neighborhood = d_nn.mean(dim=1)
        score_flip = score_mean * torch.exp(-self.lambda_ * (D_neighborhood / (score_coefficient_of_variation + eps)))

        return preds, [
            score_mean,
            score_median,
            score_mean/(score_variance+eps),
            score_mean/(score_std_deviation+eps),
            score_mean/(score_mean_absolute_deviation+eps),
            score_mean/(score_coefficient_of_variation+eps),
            score_mean/(score_coefficient_of_variation+eps)**2,
            score_mean**8/(score_coefficient_of_variation+eps),
            score_mean/(score_avg_pairwise_distance+eps),
            score_mean/(mutual_information+eps),
            score_mean/(mean_js_divergence+eps),
            score_mean*disagreement_score,
            score_mean*weight_entropy,
            score_mean*std_log_logits,
            score_final,
            score_min_max,
            score_localized,
            score_hybrid,
            score_flip,
        ]

