import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np


class AFLWAnnotationTransform(object):
    """Transforms an AFLW annotation entry into a Tensor of bbox coords and label index."""
    def __init__(self):
        pass

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): AFLW target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        res = []
        scale = np.array([width, height, width, height])
        for obj in target:
            # Get bounding box
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                bbox_list = list(np.array(bbox) / scale)
                # Get label idx
                label_idx = obj['category_id']
                bbox_list.append(label_idx)
                res += [bbox_list]
            else:
                raise RuntimeError("No bounding box found! Check annotation file - Abort.")
        return res


class AFLW(data.Dataset):
    """AFLW Dataset.
    Args:
        root (string): Root directory where images have been downloaded to.
        transform (callable, optional): A function/transform that augments the raw images.
        target_transform (callable, optional): A function/transform that takes in the target (bbox) and transforms it.
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=AFLWAnnotationTransform()):
        sys.path.append(osp.join(root, "PythonAPI"))
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(osp.join(self.root, "aflw_annotations.json"))
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, bbox_gt, h, w, img_id, img_path = self.pull_item(index)
        return im, bbox_gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        path = osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(path)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        target = np.array(target)
        if self.transform is not None:
            img, boxes, labels = self.transform(img=img, boxes=target[:, :4], labels=target[:, -1])
        else:
            boxes, labels = target[:, :4], target[:, -1]
        bbox_target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), bbox_target, height, width, img_id, path

    def pull_image(self, index):
        """ Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        """
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
