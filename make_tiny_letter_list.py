import glob
import os
import random

Class_name_list = sorted(glob.glob("./datasets/by_merge/*"))
save_train_list = './datasets/tiny_letter_gt_tr.txt'
save_test_list = './datasets/tiny_letter_gt_te.txt'

ascii_letter = {
    '42': 'B',
    '43_63': 'C',
    '44': 'D',
    '45': 'E',
    '46': 'F',
    '48': 'H',
    '49_69': 'I',
    '4c_6c': 'L',
    '54': 'T',
    '5a_7a': 'Z'
}

ascii_letterIdx = {
    '42': '0',
    '43_63': '1',
    '44': '2',
    '45': '3',
    '46': '4',
    '48': '5',
    '49_69': '6',
    '4c_6c': '7',
    '54': '8',
    '5a_7a': '9'
}

img_path_list = []
class_list = []
for index, class_num in enumerate(Class_name_list):

    class_num_tmp = class_num.split('/')
    class_num_tmp = class_num_tmp[-1]
    if class_num_tmp in ascii_letterIdx:
        sample_list = sorted(glob.glob(class_num + "/hsf_0/*"))
        for i in sample_list:
            img_path_list.append(i)
            class_list.append(ascii_letterIdx[class_num_tmp])
        sample_list = sorted(glob.glob(class_num + "hsf_1/*"))
        for i in sample_list:
            img_path_list.append(i)
            class_list.append(ascii_letterIdx[class_num_tmp])

# we only use hsf_0 and hsf_1 as training samples, others can be as testing
num_tr = 0
fl = open(save_train_list, 'w')
for i in range(len(img_path_list)):
    example_info = img_path_list[i] + " " + str(class_list[i]) + " " + str(num_tr)
    fl.write(example_info)
    fl.write("\n")
    num_tr = num_tr + 1
fl.close()

img_path_list = []
class_list = []
for index, class_num in enumerate(Class_name_list):

    class_num_tmp = class_num.split('/')
    class_num_tmp = class_num_tmp[-1]
    if class_num_tmp in ascii_letterIdx:
        sample_list = sorted(glob.glob(class_num + "/hsf_2/*"))
        for i in sample_list:
            img_path_list.append(i)
            class_list.append(ascii_letterIdx[class_num_tmp])

num_te = 0
fl = open(save_test_list, 'w')
for i in range(len(img_path_list)):
    example_info = img_path_list[i] + " " + str(class_list[i]) + " " + str(num_te)
    fl.write(example_info)
    fl.write("\n")
    num_te = num_te + 1
fl.close()

# num_tr_te = 0
# fl = open(save_train_test_list, 'w')
# for i in range(len(img_path_list)):
#     example_info = img_path_list[i] + " " + str(class_list[i]) + " " + str(num_tr_te)
#     fl.write(example_info)
#     fl.write("\n")
#     num_tr_te = num_tr_te + 1
# fl.close()
#
#
#
#
# # training testing split
# together = list(zip(img_path_list, class_list))
# random.shuffle(together)
# img_path_list, class_list = zip(*together)
#
# train_img_path_list = img_path_list[0:int(len(img_path_list)*0.8)]
# train_class_list = class_list[0:int(len(img_path_list)*0.8)]
# num_tr = 0
# fl = open(save_train_list, 'w')
# for i in range(len(train_img_path_list)):
#     example_info = train_img_path_list[i] + " " + str(train_class_list[i]) + " " + str(num_tr)
#     fl.write(example_info)
#     fl.write("\n")
#     num_tr = num_tr + 1
# fl.close()
#
# test_img_path_list = img_path_list[int(len(img_path_list)*0.8):]
# test_class_list = class_list[int(len(img_path_list)*0.8):]
# num_te = 0
# fl = open(save_test_list, 'w')
# for i in range(len(test_img_path_list)):
#     example_info = test_img_path_list[i] + " " + str(test_class_list[i]) + " " + str(num_te)
#     fl.write(example_info)
#     fl.write("\n")
#     num_te = num_te + 1
# fl.close()


