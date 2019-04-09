import argparse
import json


def main():
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser("Concatenate two COCO-style json annotation files")
    parser.add_argument('-v', '--verbose', action='store_true', help="increase output verbosity")
    parser.add_argument('-a', '--first', type=str, required=True, help="set first json annotation file")
    parser.add_argument('-b', '--second', type=str, required=True, help="set second json annotation file")
    parser.add_argument('-m', '--merged', type=str, required=True, help="set merged json annotation file")
    args = parser.parse_args()

    # Load first annotation file
    if args.verbose:
        print("#. Load first annotation file  : {}".format(args.first))

    with open(args.first, 'r') as f:
        dict_a = json.load(f)

    if args.verbose:
        print("  \\__Number of images         : {}".format(len(dict_a['images'])))
        print("  \\__Number of annotations    : {}".format(len(dict_a['annotations'])))

    # Load second annotation file
    if args.verbose:
        print("#. Load second annotation file : {}".format(args.second))

    with open(args.second, 'r') as f:
        dict_b = json.load(f)

    if args.verbose:
        print("  \\__Number of images         : {}".format(len(dict_b['images'])))
        print("  \\__Number of annotations    : {}".format(len(dict_b['annotations'])))

    # Merge annotation dictionaries
    if args.verbose:
        print("#. Merge annotation dictionaries and save at: {}".format(args.merged))

    # Build merged COCO-like dictionary
    merged_dict = dict()

    # Dataset info
    dataset_info = {
        'description': dict_a['description'] + ' | ' + dict_b['description'],
        'url': dict_a['url'] + ' | ' + dict_b['url'],
        'version': dict_a['version'] + ' | ' + dict_b['version'],
        'year': max(dict_a['year'], dict_b['year']),
        'contributor': dict_a['contributor'] + ' | ' + dict_b['contributor'],
        'date_created': dict_a['date_created'] + ' | ' + dict_b['date_created']
    }
    merged_dict.update(dataset_info)

    # Dataset licenses
    dataset_licenses = {'licenses': dict_a['licenses'] + dict_b['licenses']}
    merged_dict.update(dataset_licenses)

    # Dataset images
    dataset_images = {'images': dict_a['images'] + dict_b['images']}
    merged_dict.update(dataset_images)

    # Dataset annotations
    dataset_annotations = {'annotations': dict_a['annotations'] + dict_b['annotations']}
    merged_dict.update(dataset_annotations)

    # Dataset categories
    dataset_categories = {'categories': dict_a['categories'] + dict_b['categories']}
    merged_dict.update(dataset_categories)

    if args.verbose:
        print("  \\__Number of images         : {}".format(len(merged_dict['images'])))
        print("  \\__Number of annotations    : {}".format(len(merged_dict['annotations'])))

    with open(args.merged, 'w') as fp:
        json.dump(merged_dict, fp)


if __name__ == '__main__':
    main()
