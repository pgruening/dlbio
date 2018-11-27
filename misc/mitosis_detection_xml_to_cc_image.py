import collections
import glob
import os
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

from ..helpers import file_path_to_ID, load_image, pRectangle

#Rectangle = collections.namedtuple('Rectangle', 'x y h w')
POS_ID = 0


class MitosisXMLToCCImage():
    def __call__(self, src_folders, src_globs, dst_folder, SHOW_IMAGES=False):
        """Take a folder with images and a folder with xml files
        and convert them into 'virtual' cc_images. Of course, instance
        segmentation will not make any sense when using those labels,
        but when working with cc_images many detection functions from
        the detseg method can be used. 

        Parameters
        ----------
        src_folders : [str, str]
            first string points to image folder second one to xml files.
        src_globs : [str, dontcare]
            first glob string indicates which images are used. Second needs to
            be '*.xml'. Here, the input is not regarded.
            [description]
        dst_folder : str
            Where to save the files as .npy files.

        """

        image_folder = src_folders[0]
        image_glob = src_globs[0]

        full_image_glob = os.path.join(
            image_folder,
            image_glob
        )

        image_file_paths = glob.glob(full_image_glob)
        print('found {} images fro glob {}'.format(
            len(image_file_paths), full_image_glob
        ))

        xml_folder = src_folders[1]
        xml_glob = '*.xml'

        xml_file_paths = glob.glob(os.path.join(
            xml_folder,
            xml_glob
        ))
        print('found {} xml_files'.format(len(xml_file_paths)))

        xml_dict = {file_path_to_ID(x): x for x in xml_file_paths}

        for image_path in image_file_paths:
            image_id = file_path_to_ID(image_path)

            if image_id not in xml_dict.keys():
                print('File {} not found in xml ids'.format(image_id))
                continue

            xml_filepath = xml_dict.pop(image_id)

            gt_rectangles = load_ground_truth(xml_filepath)

            image = load_image(image_path, None)
            h, w = image.shape[0], image.shape[1]
            cc_image = np.zeros((h, w), dtype='int32')

            index = 1
            for rect in gt_rectangles:
                cc_image[rect.y:rect.y+rect.h, rect.x:rect.x+rect.w] = index
                index += 1

            if SHOW_IMAGES:
                _, ax = plt.subplots(1, 2)
                ax[0].imshow(image)
                ax[1].imshow(cc_image)

                # draw rectangles onto image
                for rect in gt_rectangles:
                    r_patch = rect.get_pyplot_patch()
                    ax[0].add_patch(r_patch)

                plt.show()

            output_path = os.path.join(
                dst_folder,
                image_id+'.npy'
            )

            np.save(output_path, cc_image)
            print('File saved to {}'.format(output_path))

        for key in xml_dict.keys():
            print('No image found for xml_file: {}'.format(key))


def load_ground_truth(file_str, load_positive_only=True):

    root = ET.parse(file_str).getroot()
    prefix = root.tag.split('annotation')[0]
    annotation_objects = root.iter(prefix+'object')

    ground_truth_rectangles = []
    for obj in annotation_objects:

        if load_positive_only and int(obj.findtext(prefix+'type_id')) != POS_ID:
            continue

        rectangle = obj.find(prefix+'rectangle')

        x = int(rectangle.findtext(prefix+'left_x'))
        y = int(rectangle.findtext(prefix+'top_y'))
        w = int(rectangle.findtext(prefix+'width'))
        h = int(rectangle.findtext(prefix+'height'))

        ground_truth_rectangles.append(pRectangle(x=x, y=y, w=w, h=h))

    return ground_truth_rectangles
