from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import  sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import src._init_paths

import os

import cv2
import numpy as np

from progress.bar import Bar
import torch
# from src.lib.external.nms import soft_nms
from src.lib.opts import opts
from src.lib.logger import Logger
from src.lib.utils.utils import AverageMeter
from src.lib.datasets.dataset_factory import dataset_factory
from src.lib.detectors.detector_factory import detector_factory
from src.lib.datasets.build import build_dataset

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.image_set_index = dataset.image_set_index
    self.pattern =  dataset.pattern
    self.frame_id =  dataset.frame_id
    self.frame_seg_id =  dataset.frame_seg_id
    self.frame_seg_len =  dataset.frame_seg_len
    self.img_dir = dataset._img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
    self.test_offsets = [int(i) for i in opt.test_offsets.split(',')]
  
  def __getitem__(self, index):

    img_id = index
    file_name = self.image_set_index[index]
    img_path = self.img_dir % file_name
    image = cv2.imread(img_path)

    #pre im
    offsets = self.test_offsets
    ref_imgs=[]
    for i in range(self.opt.ref_num):
      ref_id = min(max(self.frame_seg_id[index] + offsets[i], 0), self.frame_seg_len[index] - 1)
      ref_filename = self.pattern[index] % ref_id
      ref_path = self.img_dir % ref_filename
      ref_img = cv2.imread(ref_path)
      ref_imgs.append(ref_img)

    images, pre_images, meta = {}, {}, {}
    for scale in opt.test_scales:
      images[scale], meta[scale] = self.pre_process_func(image, scale)

      proccesed_ref_imgs = []
      for r, ref_img in enumerate(ref_imgs):
        processed_ref_img, _ = self.pre_process_func(ref_img, scale)
        processed_ref_img = torch.unsqueeze(processed_ref_img, dim=0)
        proccesed_ref_imgs.append(processed_ref_img)

      pre_images[scale] = torch.cat(proccesed_ref_imgs, dim=1)
    return img_id, {'images': images, 'image': image, 'pre_images': pre_images,
                    'pre_img': np.concatenate(ref_imgs, axis=0), 'meta': meta}

  def __len__(self):
    return len(self.image_set_index)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  # split = 'val'if not opt.trainval else 'test'
  detector = Detector(opt)

  datasets = build_dataset(Dataset, opt, is_train=False)
  datasets = datasets[0]

  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, datasets, detector.pre_process),
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

  results = {}
  num_iters = len(datasets)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()
  for t in avg_time_stats:
    print('|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(t, tm=avg_time_stats[t]))
  datasets.run_eval(results, opt.save_dir)

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)
  dataset = build_dataset(Dataset, opt, data_name=opt.dataset, task=opt.task)
  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    if opt.task == 'ddd':
      ret = detector.run(img_path, img_info['calib'])
    else:
      ret = detector.run(img_path)
    
    results[img_id] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
  opt = opts().parse()
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)