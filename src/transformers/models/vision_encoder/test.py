from transformers import ResNetConfig, VisionEncoderConfig, VisionEncoder

config = VisionEncoderConfig(ResNetConfig())

# model = VisionEncoder(config)