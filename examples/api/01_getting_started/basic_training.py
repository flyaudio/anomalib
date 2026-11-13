# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Getting Started with Anomalib Training using the Python API.

This example shows the basic steps to train an anomaly detection model
using the Anomalib Python API.
"""

# 1. Import required modules
from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore

if __name__ == "__main__":
    # 2. Create a dataset
    # MVTecAD is a popular dataset for anomaly detection
    datamodule = MVTecAD(
        root=None,  # None resolves to ~/.cache/anomalib/datasets/MVTecAD
        category="pill",  # MVTec category to use
        train_batch_size=32,  # Number of images per training batch
        eval_batch_size=32,  # Number of images per validation/test batch
    )

    # 3. Initialize the model
    # Patchcore is a good choice for beginners
    model = Patchcore(
        num_neighbors=6,  # Override default model settings
    )

    # 4. Create the training engine
    engine = Engine(
        max_epochs=1,  # Override default trainer settings
    )

    # 5. Train the model
    # This produces a lightning model (.ckpt)
    engine.fit(datamodule=datamodule, model=model)

    # 6. Test the model performance
    # Metrics (image/pixel AUROC, ...) are printed to the console and written
    # to a CSV file under results/
    test_results = engine.test(datamodule=datamodule, model=model)

    # 7. Export the model
    # Different formats are available: Torch, OpenVINO, ONNX
    # NOTE: OpenVINO export requires `pip install openvino`; comment out this
    # step to skip it.
    engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
    )
