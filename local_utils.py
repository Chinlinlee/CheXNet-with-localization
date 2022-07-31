import os
from abc import ABC

import cv2
import dicom2jpg
import highdicom
import numpy as np
import pydicom
import scipy
import skimage
import skimage.transform
import torch.nn as nn
import torchvision
from pydicom.valuerep import PersonName


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
    cv2.imwirte(full_jpeg_filename, image_data, image_quality)
    return True


pass


def create_gsps_from_bounding_box(ds: pydicom.Dataset, prediction: str, index: int):
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
            [np.round(x1), np.round(y1), np.round(x1+10), np.round(y1+20)]  # left, top, right, bottom
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
    gsps.save_as(f"gsps_{prediction_name}_{index}.dcm")

pass

# endregion
