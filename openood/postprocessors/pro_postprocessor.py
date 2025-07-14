from typing import Any
import torch
import torch.nn as nn
import numpy as np
from openood.utils.config import Config

from .base_postprocessor import BasePostprocessor

class PROPostprocessor(BasePostprocessor):
    """
    A general postprocessor that applies the Perturbation Robustness (PRO) logic
    to any given base score postprocessor.

    This version supports a combined hyperparameter sweep (APS) over its own
    parameters (gd_steps, noise_level) and the parameters of the wrapped
    base postprocessor.
    """

    def __init__(self, config):
        super().__init__(config)
        self.args = config.postprocessor.postprocessor_args

        # PRO-specific hyperparameters
        self.gd_steps = self.args.gd_steps
        self.noise_level = self.args.noise_level

        # --- Base Postprocessor Instantiation ---
        from openood.postprocessors.utils import get_postprocessor
        self.score_pp_name = self.args.score_postprocessor
        base_pp_config_path = f'configs/postprocessors/{self.score_pp_name}.yml'
        base_pp_config = Config(base_pp_config_path)
        base_pp_config.dataset = config.dataset
        self.base_pp = get_postprocessor(base_pp_config)

        # --- Combined Hyperparameter Sweep Logic ---
        self.APS_mode = config.postprocessor.APS_mode
        self.APS_score = getattr(self.args, 'APS_score', None)
        if self.APS_mode and self.APS_score is None:
            raise ValueError("APS_mode is True but APS_score not provided for PROPostprocessor")
        self.aps_score_idx = None

        # Combine hyperparameter sweep dictionaries from PRO and its base
        pro_sweep = getattr(config.postprocessor, 'postprocessor_sweep', {}) or {}
        base_sweep = getattr(self.base_pp, 'args_dict', {}) or {}
        self.args_dict = {}
        self.args_dict.update(pro_sweep)
        self.args_dict.update(base_sweep)

        # --- Metric Labels ---
        self.metric_labels = [
            'score_base',     # The original score from the base postprocessor
            'score_pro',      # The score after the final perturbation (v1 style)
            'score_pro_v2',   # The minimum score across the perturbation trajectory (v2 style)
        ]

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """Pass the setup call to the base postprocessor, if it has a setup method."""
        try:
            self.base_pp.setup(net, id_loader_dict, ood_loader_dict)
            print(f"Base postprocessor '{self.score_pp_name}' setup complete.")
        except AttributeError:
            # This is expected if the base postprocessor does not need setup
            pass

    def postprocess(self, net: nn.Module, data: Any):
        temp_inputs = data.clone().detach()
        with torch.no_grad():
            unperturbed_pred, score_base = self.base_pp.postprocess(net, temp_inputs)

        conf_record = [score_base.detach().clone()]
        
        for _ in range(self.gd_steps):
            temp_inputs.requires_grad = True
            # call unwrapped base postprocess to enable gradient tracking
            unwrapped_pp = self.base_pp.postprocess.__wrapped__
            _, conf = unwrapped_pp(self.base_pp, net, temp_inputs)
            conf_record.append(conf.detach().clone())
            loss = conf.mean()
            loss.backward()
            gradient = temp_inputs.grad.data
            temp_inputs = temp_inputs.detach()
            temp_inputs = torch.add(temp_inputs, gradient.sign(), alpha=-self.noise_level)
        
        with torch.no_grad():
            _, score_pro = self.base_pp.postprocess(net, temp_inputs)
        conf_record.append(score_pro.detach().clone())
        
        conf_record_tensor = torch.stack(conf_record, dim=0)
        score_pro_v2 = conf_record_tensor.min(dim=0).values

        return unperturbed_pred, [score_base, score_pro, score_pro_v2]

    def set_hyperparam(self, hyperparam: list):
        """
        Set hyperparameters for both the PRO logic and the wrapped base postprocessor
        from a single flat list, as required by the APS framework.
        """
        param_names = list(self.args_dict.keys())
        base_names = list(getattr(self.base_pp, 'args_dict', {}).keys())

        # Assign PRO-specific hyperparameters
        for idx, name in enumerate(param_names):
            if name not in base_names:
                val = hyperparam[idx]
                if name == 'gd_steps':
                    self.gd_steps = int(val)
                elif name == 'noise_level':
                    self.noise_level = float(val)

        # Delegate base postprocessor hyperparameters
        base_vals = [hyperparam[param_names.index(name)] for name in base_names if name in param_names]
        if base_vals:
            self.base_pp.set_hyperparam(base_vals)

    def get_hyperparam(self):
        """
        Retrieve a single flat list of hyperparameters from both PRO and its base,
        maintaining the order defined in self.args_dict.
        """
        # A robust way to get hyperparams in the correct order
        hyperparams = []
        # Get current base params as a dictionary for easy lookup
        base_param_keys = list(getattr(self.base_pp, 'args_dict', {}).keys())
        # Safely retrieve base postprocessor hyperparameters
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
            if name == 'gd_steps':
                hyperparams.append(self.gd_steps)
            elif name == 'noise_level':
                hyperparams.append(self.noise_level)
            elif name in base_params_dict:
                hyperparams.append(base_params_dict[name])
        
        return hyperparams

    def inference(self, net, data_loader, progress=True):
        """
        Perform inference. During an APS search, this will return only the specific
        score being optimized. Otherwise, it returns all computed metrics.
        """
        pred_arr, conf_arr, label_arr = super().inference(net, data_loader, progress)
        
        # If in APS mode, select only the score specified by APS_score
        if self.APS_mode and not getattr(self, 'hyperparam_search_done', False):
            # Resolve the score index if it hasn't been already
            if self.aps_score_idx is None:
                if isinstance(self.APS_score, str):
                    self.aps_score_idx = self.metric_labels.index(self.APS_score)
                else:
                    self.aps_score_idx = int(self.APS_score)
            
            if isinstance(conf_arr, np.ndarray) and conf_arr.ndim == 2:
                conf_arr = conf_arr[:, self.aps_score_idx]

        return pred_arr, conf_arr, label_arr