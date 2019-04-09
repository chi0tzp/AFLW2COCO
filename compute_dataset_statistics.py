import sys
import json
import argparse
from data import *


def progress_updt(msg, total, progress):
    bar_length, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(bar_length * progress))
    text = "\r{}[{}] {:.0f}% {}".format(msg,
                                        "#" * block + "-" * (bar_length - block),
                                        round(progress * 100, 0),
                                        status)
    sys.stdout.write(text)
    sys.stdout.flush()


def write_dict(filename, d):
    f = open(filename, "w")
    f.write(json.dumps(d))
    f.close()


def main():
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser("Compute AFLW dataset's statistics")
    parser.add_argument('-v', '--verbose', action='store_true', help="increase output verbosity")
    parser.add_argument('--dataset_root', type=str, default='./', help="set dataset's root directory")
    parser.add_argument('--mode', type=str, choices=('train', ), default='train', help="chose dataset's mode")
    args = parser.parse_args()

    # Build data loader
    dataset = AFLW(root=args.dataset_root, mode=args.mode, transform=None)

    # Total number of images in dataset
    num_images = len(dataset)

    #
    if args.verbose:
        print("#. Compute statistics for AFLW dataset...")

    img_widths = []
    img_heights = []
    bbox_widths = []
    bbox_heights = []
    bbox_diags = []
    bbox_areas = []
    bbox_labels = []
    per_channel_sum = np.zeros((1, 3))
    for i in range(num_images):
        img, gt, img_h, img_w, _, _ = dataset.pull_item(i)
        # img   : torch.Tensor
        # gt    : list of numpy.ndarray of shape (5, 1)
        # img_h : int
        # img_w : int
        img_widths.append(img_w)
        img_heights.append(img_h)
        scale = np.array([img_w, img_h, img_w, img_h])
        for bbox in gt:
            bbox_labels.append(int(bbox[-1]))
            bbox[:4] *= scale
            bbox_widths.append(bbox[2] - bbox[0])
            bbox_heights.append(bbox[3] - bbox[1])
            bbox_diags.append(np.sqrt(bbox_widths[-1] ** 2 + bbox_heights[-1] ** 2))
            bbox_areas.append(bbox_widths[-1] * bbox_heights[-1])
        per_channel_sum += img.squeeze(0).float().view(3, -1).mean(dim=1).numpy()

        if args.verbose:
            progress_updt("  \\__Processing ", num_images, i + 1)

    img_widths = np.array(img_widths)
    img_heights = np.array(img_heights)
    bbox_widths = np.array(bbox_widths)
    bbox_heights = np.array(bbox_heights)
    bbox_diags = np.array(bbox_diags)
    bbox_areas = np.array(bbox_areas)
    bbox_labels = np.array(bbox_labels)
    per_channel_mean = per_channel_sum / num_images

    if args.verbose:
        print("  \\__Image widths     : mean = {} (std={})".format(int(img_widths.mean()), int(img_widths.std())))
        print("  \\__Image heights    : mean = {} (std={})".format(int(img_heights.mean()), int(img_heights.std())))
        print("  \\__Bbox widths      : mean = {} (std={})".format(int(bbox_widths.mean()), int(bbox_widths.std())))
        print("  \\__Bbox heights     : mean = {} (std={})".format(int(bbox_heights.mean()), int(bbox_heights.std())))
        print("  \\__Bbox areas       : mean = {} (std={})".format(int(bbox_areas.mean()), int(bbox_areas.std())))
        print("  \\__Bbox diagonals   : mean = {} (std={})".format(int(bbox_diags.mean()), int(bbox_diags.std())))
        print("  \\__Per channel mean : {}".format(per_channel_mean.astype(np.int)[0]))

    # Save dictionary of dataset's statistics
    dataset_statistics_dict = {
        'img_widths': img_widths,
        'img_heights': img_heights,
        'bbox_widths': bbox_widths,
        'bbox_heights': bbox_heights,
        'bbox_areas': bbox_areas,
        'bbox_diags': bbox_diags,
        'bbox_labels': bbox_labels,
        'per_channel_mean': per_channel_mean
    }
    dataset_statistics_file = "aflw_train_statistics.npy"

    if args.verbose:
        print(".# Save dataset's statistics at {}...".format(dataset_statistics_file))

    np.save(dataset_statistics_file, dataset_statistics_dict)


if __name__ == "__main__":
    main()
