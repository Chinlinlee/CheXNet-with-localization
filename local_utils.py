import os
from abc import ABC
from io import BytesIO

import cv2
import dicom2jpg
import highdicom
import numpy as np
import pydicom
import scipy
import skimage
import skimage.transform
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset
from pydicom.valuerep import PersonName
from pydicom.filebase import DicomFileLike
from collections import OrderedDict



# model archi
# construct model
class DenseNet121(nn.Module, ABC):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    pass

    def forward(self, x):
        x = self.densenet121(x)
        return x

    pass


pass


# build test dataset
class ChestXrayDataSetPlot(Dataset):
    def __init__(self, input_x, transform=None):
        self.X = np.uint8(input_x * 255)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image
        """
        current_X = np.tile(self.X[index], 3)
        image = self.transform(current_X)
        return image
    pass

    def __len__(self):
        return len(self.X)
    pass


pass


# ======= Grad CAM Function =========
class PropagationBase(object):

    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.cuda = cuda
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

    def forward(self, image):
        self.image = image
        self.preds = self.model.forward(self.image)
        return self.preds.cpu().data.numpy()

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        if self.cuda:
            self.preds.backward(gradient=one_hot, retain_graph=True)
        else:
            self.preds.cpu().backward(gradient=one_hot, retain_graph=True)
        pass


pass


class GradCAM(PropagationBase):

    def _set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()
        pass

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()
        pass

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)
        pass
    pass

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
                    pass
                pass
            pass
        pass
        raise ValueError('Invalid layer name: {}'.format(target_layer))
    pass

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.data
    pass

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)
    pass

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = torch.FloatTensor(self.map_size).zero_()
        for fmap, weight in zip(fmaps[0], weights[0]):
            gcam += fmap * weight.data

        gcam = F.relu(Variable(gcam))

        gcam = gcam.data.cpu().numpy()
        gcam -= gcam.min()
        gcam /= gcam.max()
        gcam = cv2.resize(gcam, (self.image.size(3), self.image.size(2)))

        return gcam
    pass

    def save(self, filename, gcam, raw_image):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = gcam.astype(np.float) + raw_image.astype(np.float)
        gcam = gcam / gcam.max() * 255.0
        cv2.imwrite(filename, np.uint8(gcam))
    pass


pass


def load_and_transform_image(image_full_path: str, ds: pydicom.Dataset = None):
    """Load image and resize image to 256, 256

    Args:
        image_full_path: The full image path
        ds: pydicom dataset, use when input is .dcm file
    """
    print(f"load and transform image: {image_full_path}")
    extension_name = os.path.splitext(image_full_path)[1]
    if extension_name == ".dcm":
        convert_dicom_to_jpeg(image_full_path, ds)
        image_full_path = image_full_path.replace(".dcm", ".jpg")
    pass

    img = scipy.misc.imread(image_full_path)
    if len(img.shape) == 3:
        img = img[:, :, 0]
    pass
    img_resized = skimage.transform.resize(img, (256, 256))
    return np.array(img_resized).reshape(256, 256, 1)


pass


# region DICOM

def dicom_color_correction(i_ds, i_pixel_array):
    if 'PhotometricInterpretation' in i_ds and i_ds.PhotometricInterpretation in \
            ['YBR_RCT', 'RGB', 'YBR_ICT', 'YBR_PARTIAL_420', 'YBR_FULL_422', 'YBR_FULL', 'PALETTE COLOR']:
        i_pixel_array[:, :, [0, 2]] = i_pixel_array[:, :, [2, 0]]
    pass

    return i_pixel_array


pass


def convert_dicom_to_jpeg(full_dcm_filename, ds):
    full_jpeg_filename = full_dcm_filename.replace(".dcm", ".jpg")
    if os.path.isfile(full_jpeg_filename): return True
    image_data = dicom2jpg.dicom2img(full_dcm_filename)
    image_data = dicom_color_correction(ds, image_data)

    image_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    cv2.imwrite(full_jpeg_filename, image_data, image_quality)
    return True


pass


def create_gsps_from_bounding_box(ds: pydicom.Dataset, prediction: str, index: int, img_folder_path: str):
    original_width = ds.Columns
    original_height = ds.Rows
    scale_width = original_width / 1024
    scale_height = original_height / 1024
    prediction_name, temp_x1, temp_y1, temp_x2, temp_y2 = prediction.split(" ")

    x1, y1, x2, y2 = [0, 0, 0, 0]
    if temp_x1 < temp_x2:
        x1, y1, x2, y2 = [np.round(float(temp_x1) * scale_width, 4),
                          np.round(float(temp_y1) * scale_height, 4),
                          np.round(float(temp_x2) * scale_width, 4),
                          np.round(float(temp_y2) * scale_height, 4)]

    else:
        x1, y1, x2, y2 = [np.round(float(temp_x2) * scale_width, 4),
                          np.round(float(temp_y2) * scale_height, 4),
                          np.round(float(temp_x1) * scale_width, 4),
                          np.round(float(temp_y1) * scale_height, 4)]
    pass

    top = [x1, y1]
    right = [x2, y1]
    bottom = [x2, y2]
    left = [x1, y2]
    print(top, right, bottom, left)
    polyline = highdicom.pr.GraphicObject(
        graphic_type=highdicom.pr.GraphicTypeValues.POLYLINE,
        # Top Top->Right Right->Bottom Right-Bottom->Left
        graphic_data=np.array([
            top,
            right,
            bottom,
            left,
            top]
        ),  # coordinates of polyline vertices
        units=highdicom.pr.AnnotationUnitsValues.PIXEL,  # units for graphic data
        tracking_id=prediction_name,  # site-specific ID
        tracking_uid=highdicom.UID(),  # highdicom will generate a unique ID
    )

    # Create a text object annotation
    text = highdicom.pr.TextObject(
        text_value=prediction_name,
        bounding_box=np.array(
            [np.round(x1), np.round(y1), np.round(x1 + 10), np.round(y1 + 20)]  # left, top, right, bottom
        ),
        units=highdicom.pr.AnnotationUnitsValues.PIXEL,  # units for bounding box
        tracking_id=f"{prediction_name}_text",  # site-specific ID
        tracking_uid=highdicom.UID()  # highdicom will generate a unique ID
    )

    # Create a single layer that will contain both graphics
    # There may be multiple layers, and each GraphicAnnotation object
    # belongs to a single layer
    layer = highdicom.pr.GraphicLayer(
        layer_name='LAYER1',
        order=1,  # order in which layers are displayed (lower first)
        description='Simple Annotation Layer',
    )

    # A GraphicAnnotation may contain multiple text and/or graphic objects
    # and is rendered over all referenced images
    annotation = highdicom.pr.GraphicAnnotation(
        referenced_images=[ds],
        graphic_layer=layer,
        graphic_objects=[polyline],
        text_objects=[text]
    )
    # Assemble the components into a GSPS object
    gsps = highdicom.pr.GrayscaleSoftcopyPresentationState(
        referenced_images=[ds],
        series_instance_uid=highdicom.UID(),
        series_number=123,
        sop_instance_uid=highdicom.UID(),
        instance_number=1,
        manufacturer='CyLab',
        manufacturer_model_name='Model',
        software_versions='v1',
        device_serial_number='Device XYZ',
        content_label='ANNOTATIONS',
        institution_name='NTUNHS',
        graphic_layers=[layer],
        graphic_annotations=[annotation],
        institutional_department_name='IM',
        content_creator_name=PersonName.from_named_components(
            family_name='Lab',
            given_name='Cy'
        ),
    )

    # Save the GSPS file
    instance_uid = ds.SOPInstanceUID
    output_filename = f"gsps_{prediction_name}_{index}_{instance_uid}.dcm"
    gsps.save_as(os.path.join(img_folder_path, output_filename))
    return {
        "dataset": gsps,
        "filename": output_filename
    }


pass


def write_dataset_to_bytes(dataset):
    # create a buffer
    with BytesIO() as buffer:
        # create a DicomFileLike object that has some properties of DataSet
        memory_dataset = DicomFileLike(buffer)
        # write the dataset to the DicomFileLike object
        pydicom.dcmwrite(memory_dataset, dataset)
        # to read from the object, you have to rewind it
        memory_dataset.seek(0)
        # read the contents as bytes
        return memory_dataset.read()
    pass


pass


# endregion
