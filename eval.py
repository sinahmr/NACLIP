import argparse

from mmengine.config import Config
from mmengine.runner import Runner

import custom_datasets  # This and the following should be loaded here because of mmseg module registration
import naclip


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation with MMSeg')
    parser.add_argument('--config', default='')
    parser.add_argument('--backbone', default='')
    parser.add_argument('--arch', default='')
    parser.add_argument('--attn', default='')
    parser.add_argument('--std', default='')
    parser.add_argument('--pamr', default='')
    parser.add_argument('--work-dir', default='./work_logs/')
    parser.add_argument('--show-dir', default='', help='directory to save visualization images')
    args = parser.parse_args()
    return args


def visualization_hook(cfg, show_dir):
    if show_dir == '':
        cfg.default_hooks.pop('visualization', None)
        return
    if 'visualization' not in cfg.default_hooks:
        raise RuntimeError('VisualizationHook must be included in default_hooks, see base_config.py')
    else:
        hook = cfg.default_hooks['visualization']
        hook['draw'] = True
        visualizer = cfg.visualizer
        visualizer['save_dir'] = show_dir
        cfg.model['pamr_steps'] = 50
        cfg.model['pamr_stride'] = [1, 2, 4, 8, 12, 24]


def safe_set_arg(cfg, arg, name, func=lambda x: x):
    if arg != '':
        cfg.model[name] = func(arg)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    safe_set_arg(cfg, args.backbone, 'clip_path')
    safe_set_arg(cfg, args.arch, 'arch')
    safe_set_arg(cfg, args.attn, 'attn_strategy')
    safe_set_arg(cfg, args.std, 'gaussian_std', float)
    if args.pamr == 'off':
        cfg.model['pamr_steps'] = 0
    elif args.pamr == 'on':
        cfg.model['pamr_steps'] = 10
    visualization_hook(cfg, args.show_dir)

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
