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
from openood.evaluators.metrics import compute_all_metrics 
import numpy as np 
import inspect 

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
        # circumvent torch.no_grad in postprocess by calling the original implementation
        # pred, conf returned by the unwrapped postprocess support gradients
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
        '--eps', type=float, default=0.03, 
        help="Perturbation epsilon limit" 
    ) 
    parser.add_argument( 
        '--steps', type=int, default=40, 
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
    args = parser.parse_args() 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

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

        # setup Foolbox model and OOD criteria 
        ood_model = OODScoreModel(net, evaluator.postprocessor) 
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
                adv_images, _, _ = attack(fmodel, images, crit, epsilons=args.eps) 
                adv_list.append(torch.as_tensor(adv_images).cpu()) 
            adv_data = torch.cat(adv_list, dim=0) 
            adv_labels = torch.cat(adv_label_list, dim=0) 
            attacked_data[(split, name)] = (adv_data, adv_labels) 
            print(f"Generated {len(adv_data)} adversarial examples ({split}/{name})") 
            save_fname = f"adv_{split}_{name}_{args.attack_method}.pt" 
            save_path = os.path.join(subfolder, save_fname) 
            torch.save(adv_data, save_path) 
            print(f"Saved adversarial examples to {save_path}") 

        # Evaluate OOD performance 
        print("\n=== Evaluating OOD metrics ===", flush=True) 
        if args.ood_objective in ['min', 'minmax']: 
            # attacked ID as 'id' for evaluation 
            adv_data, adv_labels = attacked_data[('id','test')] 
            ds = DictDataset(adv_data, adv_labels) 
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4) 
            id_pred_adv, id_conf_adv, id_gt_adv = evaluator.postprocessor.inference( 
                evaluator.net, loader, progress=True) 
            id_list_adv = [id_pred_adv, id_conf_adv, id_gt_adv] 
            evaluator._eval_ood(id_list_adv, ood_split='near', progress=True) 
            evaluator._eval_ood(id_list_adv, ood_split='far', progress=True) 
        if args.ood_objective in ['max', 'minmax']: 
            # attacked OOD: combine attacked OOD with clean ID for metrics 
            # get clean ID inference 
            id_pred, id_conf, id_gt = evaluator.postprocessor.inference( 
                evaluator.net, evaluator.dataloader_dict['id']['test'], progress=True) 
            for split in ['near', 'far']: 
                pred_list, conf_list = [], [] 
                for ds_name, _ in evaluator.dataloader_dict['ood'][split].items(): 
                    adv_data, adv_labels = attacked_data[(split, ds_name)] 
                    ds = DictDataset(adv_data, adv_labels) 
                    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4) 
                    p, c, _ = evaluator.postprocessor.inference( 
                        evaluator.net, loader, progress=True) 
                    pred_list.append(p) 
                    conf_list.append(c) 
                adv_pred = np.concatenate(pred_list, axis=0) 
                adv_conf = np.concatenate(conf_list, axis=0) 
                adv_label = -1 * np.ones_like(adv_pred) 
                all_pred = np.concatenate([id_pred, adv_pred], axis=0) 
                all_conf = np.concatenate([id_conf, adv_conf], axis=0) 
                all_label = np.concatenate([id_gt, adv_label], axis=0) 
                metrics = compute_all_metrics(all_conf, all_label, all_pred) 
                print(f"\n--- OOD metrics on adversarial OOD ({split}) ---") 
                evaluator._print_metrics(metrics) 


if __name__ == "__main__": 
    main() 
