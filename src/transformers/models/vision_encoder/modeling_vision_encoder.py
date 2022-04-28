from typing import Optional

from torch import nn

from transformers import AutoConfig, AutoModel
from transformers.models.auto.configuration_auto import AutoConfig

from ...configuration_utils import PretrainedConfig
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_vision_encoder import VisionEncoderConfig


logger = logging.get_logger(__name__)


class VisionEncoder(PreTrainedModel):
    r"""
    [`VisionEncoder`] is a generic model class that will be instantiated with one of the base vision model classes of
    the library to make it act as a general feature extractor.
    """
    config_class = VisionEncoderConfig
    base_model_prefix = "backbone"
    main_input_name = "pixel_values"

    def __init__(self, config: Optional[PretrainedConfig] = None, encoder: Optional[PreTrainedModel] = None):
        if config is None and encoder is None:
            raise ValueError("Either a configuration or an encoder has to be provided.")
        if config is None:
            config = AutoConfig.from_encoder_config(encoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        # initialize with config
        super().__init__(config)

        if encoder is None:
            encoder = AutoModel.from_config(config)

        self.encoder = encoder

    def get_encoder(self):
        return self.encoder

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)

    def forward(self, pixel_values=None, **kwargs):
        outputs = self.encoder(pixel_values, output_attentions=True, **kwargs)

        hidden_states = [outputs.hidden_states[idx] for idx in self.config.out_indices]

        return hidden_states

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a specific integer. This is typically true
        for encoder / decoder type networks with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific input size divisibility is required.
        """
        return 0

    @property
    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return self.encoder.config.hidden_sizes
