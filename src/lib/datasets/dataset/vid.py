import os
import pickle

import torch.utils.data
import sys
import numpy as np
from src.lib.datasets.dataset.vid_eval import eval_detection_vid, eval_proposals_vid

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class VIDDataset(torch.utils.data.Dataset):
    classes = ['__background__',  # always index 0
                'airplane', 'antelope', 'bear', 'bicycle',
                'bird', 'bus', 'car', 'cattle',
                'dog', 'domestic_cat', 'elephant', 'fox',
                'giant_panda', 'hamster', 'horse', 'lion',
                'lizard', 'monkey', 'motorcycle', 'rabbit',
                'red_panda', 'sheep', 'snake', 'squirrel',
                'tiger', 'train', 'turtle', 'watercraft',
                'whale', 'zebra']
    classes_map = ['__background__',  # always index 0
                    'n02691156', 'n02419796', 'n02131653', 'n02834778',
                    'n01503061', 'n02924116', 'n02958343', 'n02402425',
                    'n02084071', 'n02121808', 'n02503517', 'n02118333',
                    'n02510455', 'n02342885', 'n02374451', 'n02129165',
                    'n01674464', 'n02484322', 'n03790512', 'n02324045',
                    'n02509815', 'n02411705', 'n01726692', 'n02355227',
                    'n02129604', 'n04468005', 'n01662784', 'n04530566',
                    'n02062744', 'n02391049']

    num_classes = 30
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    # alternatif
    # mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
    # std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
    def __init__(self, image_set, data_dir, img_dir, anno_path, img_index, is_train=True, opt=None):
        self.det_vid = image_set.split("_")[0]
        self.image_set = image_set
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.anno_path = anno_path
        self.img_index = img_index
        self.opt = opt

        self.is_train = is_train
        if self.is_train:
            self.split = 'train'
        self.max_objs = 128
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._img_dir = os.path.join(self.img_dir, "%s.JPEG")
        self._anno_path = os.path.join(self.anno_path, "%s.xml")

        with open(self.img_index) as f:
            lines = [x.strip().split(" ") for x in f.readlines()]
        if len(lines[0]) == 2:
            self.image_set_index = [x[0] for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
        else:
            self.image_set_index = ["%s/%06d" % (x[0], int(x[2])) for x in lines]
            self.pattern = [x[0] + "/%06d" for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
            self.frame_seg_id = [int(x[2]) for x in lines]
            self.frame_seg_len = [int(x[3]) for x in lines]

        if self.is_train:
            keep = self.filter_annotation()

            if len(lines[0]) == 2:
                self.image_set_index = [self.image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_id = [self.frame_id[idx] for idx in range(len(keep)) if keep[idx]]
            else:
                self.image_set_index = [self.image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
                self.pattern = [self.pattern[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_id = [self.frame_id[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_seg_id = [self.frame_seg_id[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_seg_len = [self.frame_seg_len[idx] for idx in range(len(keep)) if keep[idx]]

        self.classes_to_ind = dict(zip(self.classes_map, range(len(self.classes_map))))
        self.categories = dict(zip(range(len(self.classes)), self.classes))

        self.annos = self.load_annos(os.path.join(self.cache_dir, self.image_set + "_anno.pkl"))

    def __len__(self):
        return len(self.image_set_index)

    def filter_annotation(self):
        cache_file =os.path.join(self.cache_dir, self.image_set + "_keep.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                keep = pickle.load(fid)
            return keep

        keep = np.zeros((len(self)), dtype=np.bool)
        for idx in range(len(self)):
            if idx % 10000 == 0:
                print("Had filtered {} images".format(idx))

            filename = self.image_set_index[idx]

            tree = ET.parse(self._anno_path % filename).getroot()
            objs = tree.findall("object")
            keep[idx] = False if len(objs) == 0 else True
        print("Had filtered {} images".format(len(self)))

        return keep

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        objs = target.findall("object")

        for obj in objs:
            if not obj.find("name").text in self.classes_to_ind:
                continue

            bbox =obj.find("bndbox")
            box = [
                np.maximum(float(bbox.find("xmin").text), 0).astype(np.float32),
                np.maximum(float(bbox.find("ymin").text), 0).astype(np.float32),
                np.minimum(float(bbox.find("xmax").text), im_info[1] - 1).astype(np.float32),
                np.minimum(float(bbox.find("ymax").text), im_info[0] - 1).astype(np.float32)
            ]
            boxes.append(box)
            gt_classes.append(self.classes_to_ind[obj.find("name").text.lower().strip()])
        res = {
            "boxes": boxes,
            "labels": gt_classes,
            "im_info": im_info,
        }
        return res

    def load_annos(self, cache_file):
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                annos = pickle.load(fid)
        else:
            annos = []
            for idx in range(len(self)):
                if idx % 10000 == 0:
                    print("Had processed {} images".format(idx))

                filename = self.image_set_index[idx]

                tree = ET.parse(self._anno_path % filename).getroot()
                anno = self._preprocess_annotation(tree)
                annos.append(anno)
            print("Had processed {} images".format(len(self)))

        return annos

    def get_img_info(self, idx):
        im_info = self.annos[idx]["im_info"]
        return {"height": im_info[0], "width": im_info[1]}

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    @property
    def cache_dir(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_dir = os.path.join(self.data_dir, 'cache')
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        return cache_dir

    # def get_visualization(self, idx):
    #     filename = self.image_set_index[idx]
    #
    #     img = cv2.imread(self._img_dir % filename)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     target = self.get_groundtruth(idx)
    #     target = target.clip_to_image(remove_empty=True)
    #
    #     return img, target, filename

    def get_groundtruth(self, idx):
        TO_REMOVE = 1
        anno = self.annos[idx]
        targets = []
        t = len(anno['boxes'])
        w = anno['im_info'][1]
        h = anno['im_info'][0]

        for i in range(t):
            box = anno['boxes'][i]
            box[0].clip(min=0, max=w - TO_REMOVE)
            box[1].clip(min=0, max=h - TO_REMOVE)
            box[2].clip(min=0, max=w - TO_REMOVE)
            box[3].clip(min=0, max=h - TO_REMOVE)

            keep = (box[3] > box[1]) & (box[2] > box[0])
            if keep:
                cat = anno['labels'][i]
                targets.append({'bbox': box, 'category_id':cat })

        return targets

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        pred_boxlists = []
        gt_boxlists = []
        for image_id in all_bboxes:

            bbox_list = []
            cat_list = []
            score_list = []

            gt_box_list = []
            gt_cat_list = []

            for cls_ind in all_bboxes[image_id]:
                category_id = cls_ind #- 1
                for bbox in all_bboxes[image_id][cls_ind]:
                    # bbox[2] -= bbox[0]
                    # bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    score_list.append(float("{:.2f}".format(score)))
                    bbox_list.append(bbox_out)
                    cat_list.append(category_id)

            detection = {
                "image_id": int(image_id),
                "labels": np.asarray(cat_list),
                "bbox": np.asarray(bbox_list, dtype=np.float32),
                "scores": np.asarray(score_list, dtype=np.float32)
            }

            gt_boxlist = self.get_groundtruth(image_id)

            for gt in gt_boxlist:
                gt_box_list.append(gt['bbox'])
                gt_cat_list.append(gt['category_id'])

            gts = {
                "image_id": int(image_id),
                "labels": np.asarray(gt_cat_list),
                "bbox": np.asarray(gt_box_list, dtype=np.float32),
             }


            gt_boxlists.append(gts)

            pred_boxlists.append(detection)

        return pred_boxlists, gt_boxlists

    def run_eval(self, results, output_folder, box_only=False, motion_specific=True):
        pred_boxlists, gt_boxlists = self.convert_eval_format(results)

        if box_only:
            result = eval_proposals_vid(
                pred_boxlists=pred_boxlists,
                gt_boxlists=gt_boxlists,
                iou_thresh=0.5,
            )
            result_str = "Recall: {:.4f}".format(result["recall"])
            print(result_str)
            if output_folder:
                with open(os.path.join(output_folder, "proposal_result.txt"), "w") as fid:
                    fid.write(result_str)
            return

        if motion_specific:
            motion_ranges = [[0.0, 1.0], [0.0, 0.7], [0.7, 0.9], [0.9, 1.0]]
            motion_name = ["all", "fast", "medium", "slow"]
        else:
            motion_ranges = [[0.0, 1.0]]
            motion_name = ["all"]
        result = eval_detection_vid(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=0.2,
            motion_ranges=motion_ranges,
            motion_specific=motion_specific,
            use_07_metric=False
        )
        result_str = ""
        template_str = 'AP50 | motion={:>6s} = {:0.4f}\n'
        for motion_index in range(len(motion_name)):
            result_str += template_str.format(motion_name[motion_index], result[motion_index]["map"])
        result_str += "Category AP:\n"
        for i, ap in enumerate(result[0]["ap"]):
            if i == 0:  # skip background
                continue
            result_str += "{:<16}: {:.4f}\n".format(
                self.classes[i], ap
            )
        print(result_str)
        if output_folder:
            with open(os.path.join(output_folder, "result.txt"), "w") as fid:
                fid.write(result_str)

        return result

