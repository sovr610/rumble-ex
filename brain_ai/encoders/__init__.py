"""
Modality Encoders Module

Specialized encoders for vision, text, audio, and sensor inputs.
All encoders output fixed 512-dim representations for the Global Workspace.
"""

from .vision import (
    VisionEncoder,
    EventVisionEncoder,
    MultiScaleVisionEncoder,
    create_vision_encoder,
)

from .text import (
    TextEncoder,
    SpikeTextEncoder,
    CharacterTextEncoder,
    HybridTextEncoder,
    create_text_encoder,
)

from .audio import (
    AudioEncoder,
    StreamingAudioEncoder,
    MultiTaskAudioEncoder,
    MelSpectrogramFrontend,
    create_audio_encoder,
)

from .sensors import (
    SensorEncoder,
    IMUEncoder,
    MultiSensorEncoder,
    LiquidTimeConstant,
    ClosedFormContinuous,
    create_sensor_encoder,
)

from .engram_encoder import (
    EngramTextEncoder,
    create_engram_encoder,
)

__all__ = [
    # Vision
    "VisionEncoder",
    "EventVisionEncoder",
    "MultiScaleVisionEncoder",
    "create_vision_encoder",
    # Text
    "TextEncoder",
    "SpikeTextEncoder",
    "CharacterTextEncoder",
    "HybridTextEncoder",
    "create_text_encoder",
    # Audio
    "AudioEncoder",
    "StreamingAudioEncoder",
    "MultiTaskAudioEncoder",
    "MelSpectrogramFrontend",
    "create_audio_encoder",
    # Sensors
    "SensorEncoder",
    "IMUEncoder",
    "MultiSensorEncoder",
    "LiquidTimeConstant",
    "ClosedFormContinuous",
    "create_sensor_encoder",
    # Engram
    "EngramTextEncoder",
    "create_engram_encoder",
]
