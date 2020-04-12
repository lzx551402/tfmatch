#!/usr/bin/env python
"""
Copyright 2017, Zixin Luo, HKUST.
Network specifications.
"""

from cnn_wrapper.descnet import GeoDesc


class DataSpec(object):
    """Input data specifications for an ImageNet model."""

    def __init__(self,
                 batch_size,
                 input_size,
                 scale=1.,
                 central_crop_fraction=1.,
                 channels=3,
                 mean=None):
        # The recommended batch size for this model
        self.batch_size = batch_size
        # The input size of this model
        self.input_size = input_size
        # A central crop fraction is expected by this model
        self.central_crop_fraction = central_crop_fraction
        # The number of channels in the input image expected by this model
        self.channels = channels
        # The mean to be subtracted from each image. By default, the per-channel ImageNet mean.
        # ImageNet mean value: np.array([124., 117., 104.]. Values are ordered RGB.
        self.mean = mean
        # The scalar to be multiplied from each image.
        self.scale = scale


def geodesc_spec():
    """Spec for GeoDesc."""
    return DataSpec(batch_size=2,
                    input_size=(32, 32),
                    scale=0.00625,
                    central_crop_fraction=0.5,
                    channels=1,
                    mean=128)


# Collection of sample auto-generated models
MODELS = (
    GeoDesc
)

# The corresponding data specifications for the sample models
# These specifications are based on how the models were trained.
MODEL_DATA_SPECS = {
    GeoDesc: geodesc_spec(),
}


def get_data_spec(model_instance=None, model_class=None):
    """Returns the data specifications for the given network."""
    model_class = model_class or model_instance.__class__
    return MODEL_DATA_SPECS[model_class]
