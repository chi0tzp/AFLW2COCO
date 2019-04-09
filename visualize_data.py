import argparse
from data import *
import torch
import torch.utils.data as data
import numpy as np
import cv2


def main():
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser("Visualize AFLW dataset")
    parser.add_argument('-v', '--verbose', action='store_true', help="increase output verbosity")
    parser.add_argument('--dataset_root', type=str, default='./', help="set dataset's root directory")
    parser.add_argument('--mode', type=str, choices=('train', ), default='train', help="chose dataset's mode")
    parser.add_argument('--batch_size', type=int, default=1, help="set batch size")
    parser.add_argument('--dim', type=int, default=300, help="input image dimension")
    args = parser.parse_args()

    # Load AFLW dataset
    dataset = AFLW(root=args.dataset_root,
                   mode=args.mode,
                   transform=Augmentor(size=args.dim, mean=(92, 101, 113)))

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
            print(80 * "=")

        img = images[i].numpy().transpose(1, 2, 0).astype(np.uint8).copy()
        for j in range(len(bbox_targets[i])):
            # Get object's bounding box and label
            bbox = bbox_targets[i][j][:4]
            label = int(bbox_targets[i][j][-1])
            pt = (bbox * scale).numpy()

            cv2.rectangle(img, pt1=(pt[0], pt[1]), pt2=(pt[2], pt[3]), color=(255, 0, 255), thickness=2)
            if args.verbose:
                print("\tbbox = {} (label = {})".format(pt, label))

            cv2.imshow("AFLW: {}".format(i), img)
            cv2.waitKey()


if __name__ == "__main__":
    main()