import argparse
from data import *
import torch
import torch.utils.data as data
import numpy as np
import cv2


def main():
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser("Visualize AFLW dataset (COCO-style annotations)")
    parser.add_argument('-v', '--verbose', action='store_true', help="increase output verbosity")
    parser.add_argument('--dataset_root', type=str, required=True, help='AFLW root directory')
    parser.add_argument('--json', type=str, default='aflw_annotations.json', help="COCO json annotation file")
    parser.add_argument('--batch_size', type=int, default=4, help="set batch size")
    parser.add_argument('--dim', type=int, default=300, help="input image dimension")
    parser.add_argument('-a', '--augment', action='store_true', help="apply augmentations")
    args = parser.parse_args()

    # Load AFLW dataset
    if args.augment:
        transform = Augmentor(size=args.dim, mean=(92, 101, 113))
    else:
        transform = BaseTransform(size=args.dim, mean=(0, 0, 0))
    dataset = AFLW(root=args.dataset_root, transform=transform)

    # Build data loader
    data_loader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=1, shuffle=True,
                                  collate_fn=detection_collate, pin_memory=True)

    # Build batch iterator
    batch_iterator = iter(data_loader)

    try:
        images, bbox_targets = next(batch_iterator)
    except StopIteration:
        batch_iterator = iter(data_loader)
        images, bbox_targets = next(batch_iterator)

    # Get images widths and heights and compute scale
    height = images.size(2)
    width = images.size(3)
    scale = torch.Tensor([width, height, width, height])
    for i in range(args.batch_size):

        if args.verbose:
            print("Image: %d" % i)
            print("===================================================")
        
        img = images[i].numpy().transpose(1, 2, 0).astype(np.uint8).copy()
        for j in range(len(bbox_targets[i])):
            # Get object's bounding box and label
            bbox = bbox_targets[i][j][:4]
            label = int(bbox_targets[i][j][-1])
            pt = (bbox * scale).numpy()

            cv2.rectangle(img, pt1=(pt[0], pt[1]), pt2=(pt[2], pt[3]), color=(255, 0, 255), thickness=2)
            if args.verbose:
                print("\tbbox = ({}, {}, {}, {}) (label = {})".format(int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]),
                                                                      label))

            cv2.imshow("AFLW: {}".format(i), img)
            cv2.waitKey()


if __name__ == "__main__":
    main()
