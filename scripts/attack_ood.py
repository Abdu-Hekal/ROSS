#!/usr/bin/env python3 
import os 
import sys 
import argparse 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset 
from glob import glob 
from tqdm import tqdm 

import foolbox as fb 
from foolbox.criteria import OODMinConfidence, OODMaxConfidence 
import inspect 
import numpy as np
import pandas as pd

# minimal wrapper dataset for inference 
class DictDataset(Dataset): # yields {'data': x, 'label': y} 
    def __init__(self, data, labels): 
        self.data = data 
        self.labels = labels 
    def __len__(self): 
        return len(self.data) 
    def __getitem__(self, idx): 
        return {'data': self.data[idx], 'label': self.labels[idx]} 
# add project root to path 
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
sys.path.append(ROOT_DIR) 

from openood.evaluation_api.evaluator import Evaluator 
from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50 
from openood.networks.conf_branch_net import ConfBranchNet 
from openood.networks.godin_net import GodinNet 
# TODO: import other network wrappers as needed (e.g., CSINet, UDGNet, CIDERNet, etc.) 


def get_model(args): 
    """ 
    Instantiate the network architecture based on ID dataset and postprocessor. 
    """ 
    NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200} 
    MODEL_DICT = { 
        'cifar10': ResNet18_32x32, 
        'cifar100': ResNet18_32x32, 
        'imagenet200': ResNet18_224x224, 
    } 
    num_classes = NUM_CLASSES.get(args.id_data) 
    if num_classes is None: 
        raise ValueError(f"Unsupported --id-data {args.id_data}") 
    model_cls = MODEL_DICT.get(args.id_data) 
    if model_cls is None: 
        raise NotImplementedError(f"No model mapping for ID dataset {args.id_data}") 

    # instantiate backbone + head 
    if args.postprocessor == 'conf_branch': 
        net = ConfBranchNet( 
            backbone=model_cls(num_classes=num_classes), 
            num_classes=num_classes 
        ) 
    elif args.postprocessor == 'godin': 
        backbone = model_cls(num_classes=num_classes) 
        net = GodinNet( 
            backbone=backbone, 
            feature_size=backbone.feature_size, 
            num_classes=num_classes 
        ) 
    else: 
        net = model_cls(num_classes=num_classes) 

    # optional wrapper network 
    if args.wrapper_net is not None: 
        net = eval(args.wrapper_net)(backbone=net) 
    return net 


class OODScoreModel(nn.Module): 
    """ 
    Wraps a classifier and postprocessor to output OOD confidence score. 
    """ 
    def __init__(self, net, postprocessor): 
        super().__init__() 
        self.net = net 
        self.postprocessor = postprocessor 

    def forward(self, x):
        pred, conf = self.postprocessor.postprocess.__wrapped__(
                self.postprocessor, self.net, x)
        # if conf is a list or tuple, select the configured APS_score metric
        if isinstance(conf, (list, tuple)):
            idx = getattr(self.postprocessor, 'aps_score_idx', None)
            if idx is None:
                APS_score = getattr(self.postprocessor, 'APS_score', None)
                if isinstance(APS_score, str):
                    idx = self.postprocessor.metric_labels.index(APS_score)
                else:
                    idx = int(APS_score)
                self.postprocessor.aps_score_idx = idx
            conf = conf[idx]
        # convert to torch.Tensor if needed
        if not isinstance(conf, torch.Tensor):
            conf = torch.as_tensor(conf, device=x.device)
        # conf: torch.Tensor of shape (batch,)
        return conf.unsqueeze(1) 


def main(): 
    parser = argparse.ArgumentParser( 
        description="Adversarial Attack Script for OOD Detection" 
    ) 
    parser.add_argument( 
        '--root', required=True, 
        help="Root directory containing model run subfolders (e.g., s0, s1)" 
    ) 
    parser.add_argument( 
        '--postprocessor', default='msp', 
        help="Name of OOD postprocessor to attack" 
    ) 
    parser.add_argument( 
        '--id-data', type=str, default='cifar10', 
        choices=['cifar10', 'cifar100', 'imagenet200'], 
        help="In-distribution dataset name" 
    ) 
    parser.add_argument( 
        '--batch-size', type=int, default=100, 
        help="Batch size for data loader" 
    ) 
    parser.add_argument( 
        '--attack-method', type=str, default='LinfPGD', 
        help="Foolbox attack method (e.g., FGSM, LinfPGD, DeepFool)" 
    ) 
    parser.add_argument( 
        '--eps', type=float, default=0.01, 
        help="Perturbation epsilon limit" 
    ) 
    parser.add_argument( 
        '--steps', type=int, default=20, 
        help="Number of steps for iterative attacks (if applicable)" 
    ) 
    parser.add_argument( 
        '--wrapper-net', type=str, default=None, 
        help="Optional wrapper network class name, e.g., ASCOODNet" 
    ) 
    parser.add_argument( 
        '--ood-objective', choices=['min','max','minmax'], default='min', 
        help="'min': attack ID->OOD; 'max': attack OOD->ID; 'minmax': do both" 
    )
    parser.add_argument(
        '--attack-base-pp', action='store_true',
        help="Attack the base postprocessor of a composite postprocessor instead of the full one"
    )
    parser.add_argument(
        '--save-csv', action='store_true',
        help="Save CSV summary of OOD metrics"
    )
    args = parser.parse_args() 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # collect metrics across runs for summary
    all_metrics = []

    # iterate over model runs 
    for subfolder in sorted(glob(os.path.join(args.root, 's*'))): 
        print(f"Processing run: {subfolder}") 
        # load and prepare network 
        net = get_model(args) 
        ckpt_path = os.path.join(subfolder, 'best.ckpt') 
        if not os.path.isfile(ckpt_path): 
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}") 
        state = torch.load(ckpt_path, map_location='cpu') 
        net.load_state_dict(state) 
        net = net.to(device).eval() 

        # set up evaluator to get postprocessor and data loaders 
        evaluator = Evaluator( 
            net, 
            id_name=args.id_data, 
            data_root=os.path.join(ROOT_DIR, 'data'), 
            config_root=os.path.join(ROOT_DIR, 'configs'), 
            postprocessor_name=args.postprocessor, 
            postprocessor=None, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4 
        ) 
        id_loader = evaluator.dataloader_dict['id']['test']
        # determine which postprocessor to attack
        pp_to_attack = evaluator.postprocessor
        if args.attack_base_pp:
            if not hasattr(pp_to_attack, 'base_pp'):
                raise ValueError(f"Postprocessor '{args.postprocessor}' has no base_pp, cannot attack base postprocessor")
            pp_to_attack = pp_to_attack.base_pp
            print(f"Attacking base postprocessor: {pp_to_attack.__class__.__name__}")

        # setup Foolbox model and OOD criteria 
        ood_model = OODScoreModel(net, pp_to_attack) 
        ood_model = ood_model.to(device).eval() 
        fmodel = fb.PyTorchModel(ood_model, bounds=(float('-inf'), float('inf')), device=device) 
        crit_min = OODMinConfidence() 
        crit_max = OODMaxConfidence() 

        # instantiate Foolbox attack (pass steps if supported) 
        try: 
            AttackClass = getattr(fb.attacks, args.attack_method) 
        except AttributeError: 
            raise ValueError(f"Attack method '{args.attack_method}' not found in foolbox.attacks") 
        # determine if 'steps' is accepted by the attack constructor 
        sig = inspect.signature(AttackClass) 
        if 'steps' in sig.parameters: 
            attack = AttackClass(steps=args.steps) 
        else: 
            attack = AttackClass() 

        # determine which datasets to attack 
        tasks = [] 
        if args.ood_objective in ['min', 'minmax']: 
            tasks.append(('id', 'test', id_loader)) 
        if args.ood_objective in ['max', 'minmax']: 
            for split in ['near', 'far']: 
                for ds_name, ds_loader in evaluator.dataloader_dict['ood'][split].items(): 
                    tasks.append((split, ds_name, ds_loader)) 

        attacked_data = {} 
        for split, name, loader in tasks: 
            adv_list, adv_label_list = [], [] 
            for batch in tqdm(loader, desc=f"Attacking {split}/{name}"): 
                images = batch['data'].to(device) 
                labels = batch['label'].to(device) 
                adv_label_list.append(labels.cpu()) 
                # pick criterion: minimize ID or maximize OOD 
                crit = crit_min if split == 'id' else crit_max 
                try:
                    adv_images, _, _ = attack(fmodel, images, crit, epsilons=args.eps)
                except RuntimeError as e:
                    msg = str(e)
                    if 'does not require grad' in msg or 'does not have a grad_fn' in msg:
                        raise RuntimeError(
                            f"Attack method '{args.attack_method}' is gradient-based, but the postprocessor '{args.postprocessor}' output is not differentiable (no grad_fn). Please use a gradient-free attack or a postprocessor that supports autograd."
                        ) from e
                    else:
                        raise
                adv_list.append(torch.as_tensor(adv_images).cpu()) 
            adv_data = torch.cat(adv_list, dim=0) 
            adv_labels = torch.cat(adv_label_list, dim=0) 
            attacked_data[(split, name)] = (adv_data, adv_labels) 
            print(f"Generated {len(adv_data)} adversarial examples ({split}/{name})") 
            save_fname = f"adv_{split}_{name}_{args.attack_method}.pt" 
            save_path = os.path.join(subfolder, save_fname) 
            torch.save(adv_data, save_path) 
            print(f"Saved adversarial examples to {save_path}") 

        # Evaluate adversarial examples using unified eval_ood methodology
        print('\n=== Evaluating OOD metrics ===', flush=True)
        for (split, name), (adv_data, adv_labels) in attacked_data.items():
            ds_adv = DictDataset(adv_data, adv_labels)
            loader_adv = DataLoader(ds_adv, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4)
            pred_adv, conf_adv, label_adv = evaluator.postprocessor.inference(
                evaluator.net, loader_adv, progress=True)
            if split == 'id':
                evaluator.scores['id']['test'] = [pred_adv, conf_adv, label_adv]
            else:
                evaluator.scores['ood'][split][name] = [pred_adv, conf_adv, label_adv]
        # perform unified OOD evaluation using evaluator.eval_ood
        metrics_df = evaluator.eval_ood(progress=True)
        all_metrics.append(metrics_df.to_numpy())
    # after processing all runs, compute mean and std deviation across runs
    all_metrics = np.stack(all_metrics, axis=0)
    metrics_mean = np.mean(all_metrics, axis=0)
    metrics_std = np.std(all_metrics, axis=0)
    # format metrics as mean ± std
    final_metrics = []
    for i in range(metrics_mean.shape[0]):
        row = []
        for j in range(metrics_mean.shape[1]):
            row.append(f"{metrics_mean[i,j]:.2f} ± {metrics_std[i,j]:.2f}")
        final_metrics.append(row)
    df_final = pd.DataFrame(final_metrics, index=metrics_df.index, columns=metrics_df.columns)
    if args.save_csv:
        saving_root = os.path.join(args.root, 'attack_ood')
        os.makedirs(saving_root, exist_ok=True)
        csv_path = os.path.join(saving_root, f'{args.postprocessor}_{args.attack_method}.csv')
        with open(csv_path, 'w') as f:
            confs = df_final.index.get_level_values('conf').unique()
            for conf in confs:
                f.write(f"{conf or 'default'}\n")
                df_conf = df_final.xs(conf, level='conf')
                df_conf.to_csv(f)
                f.write("\n")
        print(f"Saved CSV metrics to {csv_path}")
    # print mean±std metrics per confidence method
    confs = df_final.index.get_level_values('conf').unique()
    for conf in confs:
        print(f"=== {conf or 'default'} ===")
        df_conf = df_final.xs(conf, level='conf')
        print(df_conf)
        print()


if __name__ == "__main__": 
    main() 
