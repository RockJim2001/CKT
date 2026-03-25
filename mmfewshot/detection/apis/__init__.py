# Copyright (c) OpenMMLab. All rights reserved.
from .inference import (inference_detector, init_detector,
                        process_support_images)
from .test import (multi_gpu_model_init, multi_gpu_test, single_gpu_model_init,
                   single_gpu_test)
from .train import train_detector
from .test_withcolor import single_gpu_test

__all__ = [
    'train_detector', 'single_gpu_model_init', 'multi_gpu_model_init',
    'single_gpu_test', 'multi_gpu_test', 'inference_detector', 'init_detector',
    'process_support_images', 'single_gpu_test'
]
