# modify from denseNet_localization.py
# input single DICOM file and output multiple GSPS label DICOM files in input directory

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import torch.nn as nn

import torchvision.transforms as transforms
from torch.autograd import Variable

import cv2
import os

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation

import local_utils
from local_utils import DenseNet121, ChestXrayDataSetPlot, GradCAM
import pydicom

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
torch.device('cpu')
is_dcm = False
ds = None


def inference(i_filename: str):
    """Get inference output
    Args:
        i_filename: The full path of filename
    """
    global ds
    img_folder_path, filename = os.path.split(i_filename)
    test_x = []
    gsps_dataset_list = []
    extension_name = os.path.splitext(i_filename)[1]
    if extension_name == ".dcm":
        ds = pydicom.dcmread(i_filename, stop_before_pixels=True)
    pass
    loaded_image_np = local_utils.load_and_transform_image(i_filename, ds=ds)
    test_x.append(loaded_image_np)
    test_x = np.array(test_x)

    model = DenseNet121(8)
    model = torch.nn.DataParallel(model)
    state_dict = torch.load(
        os.path.join(CURRENT_FILE_PATH, "model/DenseNet121_aug4_pretrain_WeightBelow1_1_0.829766922537.pkl"), 
        map_location="cpu"
    )
    print("model loaded")

    dataset_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = ChestXrayDataSetPlot(input_x=test_x, transform=dataset_transforms)

    thresholds = np.load(
        os.path.join(CURRENT_FILE_PATH, "thresholds.npy")
    )
    print("activate threshold", thresholds)

    print("generate heatmap ..........")
    heatmap_output, image_id, output_class = create_heatmap(model, test_dataset, thresholds)
    prediction_dict = plot_bounding_box(image_id, output_class, heatmap_output)
    for index, p in enumerate(prediction_dict[0]):
        gsps_obj = local_utils.create_gsps_from_bounding_box(ds, p, index + 1, img_folder_path)
        gsps_dataset_list.append(gsps_obj)
    pass
    return gsps_dataset_list


pass


def create_heatmap(model: nn.Module, test_dataset: Dataset, thresholds):
    # ======== Create heatmap ===========

    heatmap_output = []
    image_id = []
    output_class = []

    gcam = GradCAM(model=model, cuda=False)
    for index in range(len(test_dataset)):
        input_img = Variable((test_dataset[index]).unsqueeze(0), requires_grad=True)
        probs = gcam.forward(input_img)

        activate_classes = np.where((probs > thresholds)[0] == True)[0]  # get the activated class
        for activate_class in activate_classes:
            gcam.backward(idx=activate_class)
            output = gcam.generate(target_layer="module.densenet121.features.denseblock4.denselayer16.conv2")
            #### this output is heatmap ####
            if np.sum(np.isnan(output)) > 0:
                print("fxxx nan")
            heatmap_output.append(output)
            image_id.append(index)
            output_class.append(activate_class)
        print("test ", str(index), " finished")

    print("heatmap output done")
    print("total number of heatmap: ", len(heatmap_output))
    return heatmap_output, image_id, output_class


pass


def plot_bounding_box(image_id, output_class, heatmap_output):
    # ======= Plot bounding box =========
    img_width, img_height = 224, 224
    img_width_exp, img_height_exp = 1024, 1024

    crop_del = 16
    rescale_factor = 4

    class_index = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                   'Pneumothorax']
    avg_size = np.array([[411.8, 512.5, 219.0, 139.1], [348.5, 392.3, 479.8, 381.1],
                         [396.5, 415.8, 221.6, 318.0], [394.5, 389.1, 294.0, 297.4],
                         [434.3, 366.7, 168.7, 189.8], [502.4, 458.7, 71.9, 70.4],
                         [378.7, 416.7, 276.5, 304.5], [369.3, 209.4, 198.9, 246.0]])

    prediction_dict = {0: []}
    for img_id, k, npy in zip(image_id, output_class, heatmap_output):

        data = npy

        # output avgerge
        prediction_sent = '%s %.1f %.1f %.1f %.1f' % (
            class_index[k], avg_size[k][0], avg_size[k][1], avg_size[k][2], avg_size[k][3])
        prediction_dict[img_id].append(prediction_sent)

        if np.isnan(data).any():
            continue

        w_k, h_k = (avg_size[k][2:4] * (256 / 1024)).astype(np.int64)

        # Find local maxima
        neighborhood_size = 100
        threshold = .1

        data_max = filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        for _ in range(5):
            maxima = binary_dilation(maxima)

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects + 1)))

        for pt in xy:
            if data[int(pt[0]), int(pt[1])] > np.max(data) * .9:
                upper = int(max(pt[0] - (h_k / 2), 0.))
                left = int(max(pt[1] - (w_k / 2), 0.))

                right = int(min(left + w_k, img_width))
                lower = int(min(upper + h_k, img_height))

                prediction_sent = '%s %.1f %.1f %.1f %.1f' % (class_index[k], (left + crop_del) * rescale_factor,
                                                              (upper + crop_del) * rescale_factor,
                                                              (right - left) * rescale_factor,
                                                              (lower - upper) * rescale_factor)

                prediction_dict[img_id].append(prediction_sent)
    return prediction_dict


pass


def visualize_prediction_image(p, image_path):
    frame = cv2.imread(image_path)
    prediction_name, temp_x1, temp_y1, temp_x2, temp_y2 = p.split(" ")
    if temp_x1 < temp_x2:
        x1, y1, x2, y2 = [np.round(float(temp_x1)),
                          np.round(float(temp_y1)),
                          np.round(float(temp_x2)),
                          np.round(float(temp_y2))]

    else:
        x1, y1, x2, y2 = [np.round(float(temp_x2)),
                          np.round(float(temp_y2)),
                          np.round(float(temp_x1)),
                          np.round(float(temp_y1))]
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)  # rgb 220,20,60
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


pass

