import glob
import os
import random

Class_name_list = sorted(glob.glob("./datasets/Chinese_char2/*"))
save_train_list = './datasets/Chinese_char2_gt_tr.txt'
save_test_list = './datasets/Chinese_char2_gt_te.txt'
save_train_test_list = './datasets/Chinese_char2_gt_tr_te.txt'

img_path_list = []
class_list = []
for index, class_num in enumerate(Class_name_list):
    sample_list = sorted(glob.glob(class_num + "/*"))
    for i in sample_list:
        img_path_list.append(i)
        class_list.append(index)



num_tr_te = 0
fl = open(save_train_test_list, 'w')
for i in range(len(img_path_list)):
    example_info = img_path_list[i] + " " + str(class_list[i]) + " " + str(num_tr_te)
    fl.write(example_info)
    fl.write("\n")
    num_tr_te = num_tr_te + 1
fl.close()




# training testing split
together = list(zip(img_path_list, class_list))
random.shuffle(together)
img_path_list, class_list = zip(*together)

train_img_path_list = img_path_list[0:int(len(img_path_list)*0.8)]
train_class_list = class_list[0:int(len(img_path_list)*0.8)]
num_tr = 0
fl = open(save_train_list, 'w')
for i in range(len(train_img_path_list)):
    example_info = train_img_path_list[i] + " " + str(train_class_list[i]) + " " + str(num_tr)
    fl.write(example_info)
    fl.write("\n")
    num_tr = num_tr + 1
fl.close()

test_img_path_list = img_path_list[int(len(img_path_list)*0.8):]
test_class_list = class_list[int(len(img_path_list)*0.8):]
num_te = 0
fl = open(save_test_list, 'w')
for i in range(len(test_img_path_list)):
    example_info = test_img_path_list[i] + " " + str(test_class_list[i]) + " " + str(num_te)
    fl.write(example_info)
    fl.write("\n")
    num_te = num_te + 1
fl.close()


