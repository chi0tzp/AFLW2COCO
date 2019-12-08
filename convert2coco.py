import argparse
import sys
import os
import os.path as osp
import sqlite3
import json
import re
from PIL import Image

# Number of facial landmarks provided by AFLW dataset
N_LANDMARK = 21


def get_img_size(image_filename):
    im = Image.open(image_filename)
    return im.size[0], im.size[1]


def exec_sqlite_query(cursor, select_str, from_str=None, where_str=None):
    query_str = 'SELECT {}'.format(select_str)
    query_str += ' FROM {}'.format(from_str)
    if where_str:
        query_str += ' WHERE {}'.format(where_str)
    return [row for row in cursor.execute(query_str)]


def progress_updt(msg, total, progress):
    bar_length, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(bar_length * progress))
    text = "\r{}[{}] {:.0f}% {}".format(msg, "#" * block + "-" * (bar_length - block), round(progress * 100, 0), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def main():
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser("Convert AFLW dataset's annotation into COCO json format")
    parser.add_argument('-v', '--verbose', action="store_true", help="increase output verbosity")
    parser.add_argument('--dataset_root', type=str, required=True, help='AFLW root directory')
    parser.add_argument('--json', type=str, default='aflw_annotations.json', help="output COCO json annotation file")
    args = parser.parse_args()

    # Get absolute path of dataset root dir
    args.dataset_root = osp.abspath(args.dataset_root)

    if args.verbose:
        print("#. Transform AFLW annotations into COCO json format...")

    # Open the original AFLW annotation (sqlite database)
    if args.verbose:
        print("  \\__Open the AFLW SQLight database...", end="")
        sys.stdout.flush()

    conn = sqlite3.connect(osp.join(args.dataset_root, 'aflw.sqlite'))
    cursor = conn.cursor()
    if args.verbose:
        print("Done!")

    # Build sqlite queries
    select_str = "faces.face_id, " \
                 "imgs.filepath, " \
                 "rect.x, rect.y, " \
                 "rect.w, " \
                 "rect.h, " \
                 "pose.roll, " \
                 "pose.pitch, " \
                 "pose.yaw, " \
                 "metadata.sex"
    from_str = "faces, " \
               "faceimages " \
               "imgs, " \
               "facerect rect, " \
               "facepose pose, " \
               "facemetadata metadata"
    where_str = "faces.file_id = imgs.file_id and " \
                "faces.face_id = rect.face_id and " \
                "faces.face_id = pose.face_id and " \
                "faces.face_id = metadata.face_id"
    query_res = exec_sqlite_query(cursor, select_str, from_str, where_str)

    # Count total number of images in AFLW dataset
    if args.verbose:
        print("  \\__Count total number of images in AFLW database: ", end="")
        sys.stdout.flush()
    total_num_images = 0
    for _ in query_res:
        total_num_images += 1
    if args.verbose:
        print(total_num_images)

    # Output file for appending the file paths of not found images
    not_found_images_file = 'not_found_images_aflw.txt'
    try:
        os.remove(not_found_images_file)
    except OSError:
        pass

    # Temporary dataset variables
    aflw_dataset_dict = dict()

    # Register to dataset_dict
    img_cnt = 0
    for face_id, path, rectx, recty, rectw, recth, roll, pitch, yaw, gender in query_res:

        img_cnt += 1

        # Get current image path
        img_path = osp.join(args.dataset_root, 'flickr', path)

        # Process current image
        if osp.isfile(img_path):
            img_w, img_h = get_img_size(img_path)

            keypoints = N_LANDMARK * 3 * [0]
            pose = [roll, pitch, yaw]
            gender = 0 if gender == 'm' else 1

            # Register
            aflw_dataset_dict[face_id] = {
                'face_id': face_id,
                'img_path': osp.join('flickr', path),
                'width': img_w,
                'height': img_h,
                'bbox': (rectx, recty, rectw, recth),
                'keypoints': keypoints,
                'pose': pose,
                'gender': gender}

        # If current image file does not exist append the not found images filepaths to `not_found_images_file` and
        # continue with the next image file.
        else:
            with open(not_found_images_file, "a") as out:
                out.write("%s\n" % img_path)
            continue

        # Show progress bar
        if args.verbose:
            progress_updt("  \\__Populate AFLW dataset dictionary...", total_num_images, img_cnt)

    if args.verbose:
        print("  \\__Update AFLW dataset dictionary with keypoints...", end="")
        sys.stdout.flush()

    # Landmark property
    # (Visibility is expressed by lack of the coordinate's row.)
    select_str = "faces.face_id, coords.feature_id, coords.x, coords.y"
    from_str = "faces, featurecoords coords"
    where_str = "faces.face_id = coords.face_id"
    query_res = exec_sqlite_query(cursor, select_str, from_str, where_str)

    # Register to dataset_dict
    invalid_face_ids = list()
    for face_id, feature_id, x, y in query_res:
        assert (1 <= feature_id <= N_LANDMARK)
        if face_id in aflw_dataset_dict:
            idx = feature_id - 1
            aflw_dataset_dict[face_id]['keypoints'][3 * idx] = x
            aflw_dataset_dict[face_id]['keypoints'][3 * idx + 1] = y
            aflw_dataset_dict[face_id]['keypoints'][3 * idx + 2] = 1

        elif face_id not in invalid_face_ids:
            invalid_face_ids.append(face_id)

    if args.verbose:
        print("Done!")

    # Close database
    if args.verbose:
        print("  \\__Close the AFLW SQLight database...", end="")
        sys.stdout.flush()
    cursor.close()
    if args.verbose:
        print("Done!")

    # Close database
    if args.verbose:
        print("  \\__Convert to COCO format...", end="")
        sys.stdout.flush()

    images_list = []
    annotations_list = []
    for face_id, face_ann in aflw_dataset_dict.items():
        img_dir_num = int(face_ann['img_path'].split("/")[1])
        img_file_num = int(re.findall(r'\d+', face_ann['img_path'].split("/")[-1].split(".")[0])[0])
        image_id = int("%d%05d" % (img_dir_num, img_file_num))

        images_list.append({'id': image_id,
                            'file_name': face_ann['img_path'],
                            'height': face_ann['height'],
                            'width': face_ann['width'],
                            'date_captured': '',
                            'flickr_url': '',
                            'license': 1,
                            'dataset': 'aflw'})

        annotations_list.append({'id': face_id,
                                 'image_id': image_id,
                                 'segmentation': [],
                                 'num_keypoints': len(face_ann['keypoints']),
                                 'area': 0,
                                 'iscrowd': 0,
                                 'keypoints': face_ann['keypoints'],
                                 'bbox': face_ann['bbox'],
                                 'category_id': 0})

    # Build COCO-like dictionary
    dataset_dict = dict()

    # =============================== Dataset Info =============================== #
    dataset_info = {
        'description': 'Annotated Facial Landmarks in the Wild (AFLW)',
        'url': 'https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/',
        'version': '1.0',
        'year': 2011,
        'contributor': '',
        'date_created': '2011'
    }
    dataset_dict.update(dataset_info)

    # ============================= Dataset Licenses ============================= #
    dataset_licenses = {
        'licenses': [
            {'id': 0,
             'url': 'https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/',
             'name': 'aflw_license'}
        ]
    }
    dataset_dict.update(dataset_licenses)

    # ============================== Dataset Images ============================== #
    dataset_images = {'images': images_list}
    dataset_dict.update(dataset_images)

    # =========================== Dataset Annotations ============================ #
    dataset_annotations = {'annotations': annotations_list}
    dataset_dict.update(dataset_annotations)

    # ============================ Dataset Categories ============================ #
    dataset_categories = {
        'categories':
            [
                {'supercategory': 'face',
                 'name': 'face',
                 'skeleton': [],
                 'keypoints': ['LeftBrowLeftCorner',
                               'LeftBrowCenter',
                               'LeftBrowRightCorner',
                               'RightBrowLeftCorner',
                               'RightBrowCenter',
                               'RightBrowRightCorner',
                               'LeftEyeLeftCorner',
                               'LeftEyeCenter',
                               'LeftEyeRightCorner',
                               'RightEyeLeftCorner',
                               'RightEyeCenter',
                               'RightEyeRightCorner',
                               'LeftEar',
                               'NoseLeft',
                               'NoseCenter',
                               'NoseRight',
                               'RightEar',
                               'MouthLeftCorner',
                               'MouthCenter',
                               'MouthRightCorner',
                               'ChinCenter'],
                 'id': 0}
            ]
    }
    dataset_dict.update(dataset_categories)

    if args.verbose:
        print("Done!")

    # Save dataset dictionary as json file
    if args.verbose:
        print("  \\__Save dataset dictionary as json file...", end="")
        sys.stdout.flush()

    with open(args.json, 'w') as fp:
        json.dump(dataset_dict, fp)

    if args.verbose:
        print("Done!")


if __name__ == "__main__":
    main()
