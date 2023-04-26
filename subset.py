# create a subset of files in a folder
import argparse
import glob
import os
import numpy as np
import pdb

# for repeatability
ratio = 0.25
np.random.seed(42)

if __name__ == "__main__":

    # get path from cmd parameter

    # pdb.set_trace()

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--pathimage", required=True, help="path to folder")
    ap.add_argument("-t", "--pathtarget", required=True, help="path to folder")
    args = vars(ap.parse_args())

    root = args["pathimage"]
    root2 = args["pathtarget"]
    os.makedirs(os.path.join(root, "subset"), exist_ok=True)
    os.makedirs(os.path.join(root2, "subset"), exist_ok=True)

    man_set_ratio = int(input("set the percentage of the subset (eg. 25): "))

    # set ratio if man_set_ratio is valid
    if 100 > man_set_ratio > 0:
        ratio = man_set_ratio/100

    # get all files of certain type
    f_types = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif')
    all_files_images = []
    all_files_target = []
    for f_type in f_types:
        all_files_images.extend(sorted(glob.glob(os.path.join(root, f_type))))
        all_files_target.extend(sorted(glob.glob(os.path.join(root2, f_type))))

    # get random idx
    randomIdx = np.random.permutation(len(all_files_images))
    files_sel_idx = randomIdx[0:int(ratio * len(all_files_images))]

    for n, idx in enumerate(files_sel_idx):
        # os.system('cls')
        print(int(n/len(files_sel_idx)*10000)/100)
        os.rename(all_files_images[idx], os.path.join(root, "subset", os.path.basename(all_files_images[idx])))
        os.rename(all_files_target[idx], os.path.join(root2, "subset", os.path.basename(all_files_target[idx])))

        '''
        # move potentially existing label file
        if os.path.exists(all_files[idx][:-3] + "xml"):
            os.rename(all_files[idx][:-3] + "xml", root + "\\subset\\" + os.path.basename(all_files[idx][:-3] + "xml"))
        if os.path.exists(all_files[idx][:-3] + "json"):
            os.rename(all_files[idx][:-3] + "json", root + "\\subset\\" + os.path.basename(all_files[idx][:-3] + "json"))
        if os.path.exists(all_files[idx][:-3] + "txt"):
            os.rename(all_files[idx][:-3] + "txt", root + "\\subset\\" + os.path.basename(all_files[idx][:-3] + "txt"))
        '''