import sys
import os
from mmcv import Config

from mmdet.ppal.sampler import  *
from mmdet.ppal.builder import builder_al_sampler
from mmdet.ppal.utils.running_checks import (
    display_latest_results,
    command_with_time,
    sys_echo
)

import argparse
parser = argparse.ArgumentParser(description='Active learning arguments')
parser.add_argument('--config', required=True, type=str, help='active learning config')
parser.add_argument('--resume', required=False, type=bool, default=False, help='whether to resume training')
parser.add_argument('--model', required=True, type=str, help='running model')
args = parser.parse_args()

cfg = Config.fromfile(args.config)

sys_echo('>> Start Pascal VOC active learning')
sys_echo('>> Working path: %s' % cfg.get('output_dir'))
sys_echo('>> Config: %s' % args.config)
sys_echo('\n')

PYTHON = cfg.get('python_path', 'python')

def get_start_round():
    start_round = 0
    if args.resume:
        if not os.path.isdir(cfg.get('output_dir')):
            pass  # output_dir does not exist, starting from scratch
        else:
            k = 0
            while k < cfg.get('round_num'):
                round_work_dir = os.path.join(cfg.get('output_dir'), 'round%d' % (k + 1))
                if os.path.isfile(os.path.join(round_work_dir, 'annotations', 'new_labeled.json')):
                    k += 1
                else:
                    break
            start_round = k
    return start_round

uncertainty_sampler = builder_al_sampler(cfg.uncertainty_sampler_config)
diversity_sampler   = builder_al_sampler(cfg.diversity_sampler_config)

def run(round, run_al):
    last_round_work_dir                     = os.path.join(cfg.get('output_dir'), 'round%d'%(round-1))
    round_work_dir                          = os.path.join(cfg.get('output_dir'), 'round%d'%round)
    round_labeled_json                      = os.path.join(round_work_dir, 'annotations', 'labeled.json')
    round_unlabeled_json                    = os.path.join(round_work_dir, 'annotations', 'unlabeled.json')
    round_eval_log                          = os.path.join(round_work_dir, 'eval.txt')

    round_uncertainty_inference_json_prefix = os.path.join(round_work_dir, 'unlabeled_inference_result')
    round_uncertainty_inference_json        = os.path.join(round_work_dir, 'unlabeled_inference_result.bbox.json')
    round_uncertainty_new_labeled_json      = os.path.join(round_work_dir, 'annotations', 'uncertainty_new_labeled.json')
    round_uncertainty_new_unlabeled_json    = os.path.join(round_work_dir, 'annotations', 'uncertainty_new_unlabeled.json')

    round_diversity_image_dis_npy           = os.path.join(round_work_dir, 'image_dis.npy')
    round_diversity_inference_json_prefix   = os.path.join(round_work_dir, 'diversity_inference_result')
    round_diversity_inference_json          = os.path.join(round_work_dir, 'diversity_inference_result.bbox.json')
    round_diversity_new_labeled_json        = os.path.join(round_work_dir, 'annotations', 'new_labeled.json')
    round_diversity_new_unlabeled_json      = os.path.join(round_work_dir, 'annotations', 'new_unlabeled.json')

    train_command = '%s -m torch.distributed.launch '%PYTHON + \
                    ' --nproc_per_node=%d ' % int(cfg.get('gpus')) + \
                    ' --master_port=%d ' % int(cfg.get('port')) + \
                    ' tools/train.py ' + \
                    ' %s ' % cfg.get('train_config') + \
                    ' --work-dir %s ' % round_work_dir + \
                    ' --launcher pytorch ' + \
                    ' --cfg-options labeled_data=%s unlabeled_data=%s data.train.ann_file=%s' % (round_labeled_json, round_unlabeled_json, round_labeled_json)

    eval_command  = '%s -m torch.distributed.launch '%PYTHON + \
                   ' --nproc_per_node=%d ' % int(cfg.get('gpus')) + \
                   ' --master_port=%d ' % int(cfg.get('port')) + \
                   ' tools/test.py ' + \
                   ' %s ' % cfg.get('train_config') + \
                   ' %s ' % os.path.join(round_work_dir, 'latest.pth') + \
                   ' --work-dir %s ' % round_work_dir + \
                   ' --launcher pytorch ' + \
                   ' --eval mAP ' + \
                   ' > %s' % round_eval_log

    unlabeled_infer_command = '%s -m torch.distributed.launch '%PYTHON + \
                              ' --nproc_per_node=%d ' % int(cfg.get('gpus')) + \
                              ' --master_port=%d ' % int(cfg.get('port')) + \
                              ' tools/test.py ' + \
                              ' %s ' % cfg.get('uncertainty_infer_config') + \
                              ' %s ' % os.path.join(round_work_dir, 'latest.pth') + \
                              ' --work-dir %s ' % round_work_dir + \
                              ' --launcher pytorch ' + \
                              ' --format-only ' + \
                              ' --eval-options \"jsonfile_prefix=%s\"' % round_uncertainty_inference_json_prefix +\
                              ' --cfg-options unlabeled_data=%s data.test.ann_file=%s' % (round_unlabeled_json, round_unlabeled_json)

    if args.model == 'fasterrcnn':
        head = 'roi_head'
    else:
        head = 'bbox_head'

    os.system('mkdir -p %s' % os.path.join(round_work_dir, 'annotations'))
    if round == 1:
        os.system('cp %s %s' % (cfg.get('init_label_json'), round_labeled_json))
        os.system('cp %s %s' % (cfg.get('init_unlabeled_json'), round_unlabeled_json))
        if cfg.get('init_model', None) is not None:
            os.system('cp %s %s'%(cfg.get('init_model'),os.path.join(round_work_dir, 'latest.pth')))
        else:
            if not (args.resume and os.path.isfile(os.path.join(round_work_dir, 'latest.pth'))):
                command_with_time(train_command, 'Training')
        if cfg.get('init_inference_results', None) is not None:
            os.system('cp %s %s'%(cfg.get('init_inference_results'), round_uncertainty_inference_json))
    else:
        if args.resume and os.path.isfile(os.path.join(round_work_dir, 'latest.pth')) :
            pass
        else:
            os.system('cp %s %s' % (os.path.join(last_round_work_dir, 'annotations', 'new_labeled.json'), round_labeled_json))
            os.system('cp %s %s' % (os.path.join(last_round_work_dir, 'annotations', 'new_unlabeled.json'), round_unlabeled_json))
            command_with_time(train_command, 'Training')

    if not (os.path.isfile(round_eval_log) and args.resume):
        command_with_time(eval_command, 'Evaluation round %d'%round)
    display_latest_results(cfg.get('output_dir'), round, os.path.join(cfg.get('output_dir'), 'eval_results.txt'))

    if run_al:
        if not (os.path.isfile(round_uncertainty_inference_json) and args.resume):
            command_with_time(unlabeled_infer_command, 'Inference on unlabeled data')
        if not (os.path.isfile(round_uncertainty_new_labeled_json) and args.resume):
            uncertainty_sampler.al_round(round_uncertainty_inference_json, round_labeled_json, round_uncertainty_new_labeled_json, round_uncertainty_new_unlabeled_json)

        if hasattr(uncertainty_sampler, 'get_pool_size'):
            pool_size_round = uncertainty_sampler.get_pool_size(round + 1)
        else:
            pool_size_round = int(uncertainty_sampler.n_images)

        diversity_infer_command = '%s -m torch.distributed.launch ' % PYTHON + \
                                  ' --nproc_per_node=%d ' % int(cfg.get('gpus')) + \
                                  ' --master_port=%d ' % int(cfg.get('port')) + \
                                  ' tools/test.py ' + \
                                  ' %s ' % cfg.get('diversity_infer_config') + \
                                  ' %s ' % os.path.join(round_work_dir, 'latest.pth') + \
                                  ' --work-dir %s ' % round_work_dir + \
                                  ' --launcher pytorch ' + \
                                  ' --format-only ' + \
                                  ' --eval-options \"jsonfile_prefix=%s\"' % round_diversity_inference_json_prefix + \
                                  ' --cfg-options unlabeled_data=%s data.test.ann_file=%s' % (round_uncertainty_new_labeled_json, round_uncertainty_new_labeled_json) + \
                                  ' model.%s.total_images=%d ' % (head, pool_size_round) + \
                                  ' model.%s.output_path=\"%s\" ' % (head, round_diversity_image_dis_npy)

        if not (os.path.isfile(round_diversity_image_dis_npy) and args.resume):
            command_with_time(diversity_infer_command, 'Inference on diversity data')
        if not (os.path.isfile(round_diversity_new_labeled_json) and args.resume):
            diversity_sampler.al_round(round_uncertainty_inference_json, round_diversity_image_dis_npy, round_labeled_json, round_diversity_new_labeled_json, round_diversity_new_unlabeled_json)

        # delete inference results because they are too large
        os.system('rm -f %s' % round_uncertainty_inference_json)
        os.system('rm -f %s' % round_diversity_inference_json)
        os.system('rm -f %s' % round_diversity_image_dis_npy)


if __name__ == '__main__':
    start_round = get_start_round()
    os.system('mkdir -p %s' % cfg.get('output_dir'))
    os.system('cp %s %s' % (args.config, os.path.join(cfg.get('output_dir'), os.path.split(args.config)[-1])))
    uncertainty_sampler.set_round(start_round+1)
    diversity_sampler.set_round(start_round + 1)
    for i in range(start_round, int(cfg.get('round_num'))):
        run(i+1, i!=int(cfg.get('round_num'))-1)