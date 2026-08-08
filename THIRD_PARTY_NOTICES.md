# Third-Party Notices

## RapidOcrOnnx

Source: https://github.com/RapidAI/RapidOcrOnnx
Commit: abd498c13a6dbe5f3b3c0d421d72e01bb3e6ee6d (2025-03-25)
License: Apache-2.0 (see vendor/rapidocronnx/LICENSE-Apache-2.0.txt)

arboOCR vendors and adapts DBNet box-decoding logic (including the Clipper
polygon-offset library), CRNN CTC-decoding logic, and AngleNet
classification logic from RapidOcrOnnx. The `Detector`, `Recognizer`, and
`Classifier` classes are near-verbatim ports of RapidOcrOnnx's `DbNet`,
`CrnnNet`, and `AngleNet`.

## Clipper (via RapidOcrOnnx)

Source: bundled in RapidOcrOnnx, originally by Angus Johnson
License: Boost Software License 1.0

## PP-OCRv6 model weights

Source: https://github.com/PaddlePaddle/PaddleOCR
Redistribution: https://github.com/ARBO-TEAM/arbo-ocr-models (release assets, tag `models-v1`)
License: Apache-2.0

The `.onnx` files arboOCR downloads by default are paddle2onnx conversions
of PaddlePaddle's released PP-OCRv6 inference models. They are hosted as
release assets under the `models-v1` tag of the repository above; arboOCR
pins that tag and a SHA-256 per file rather than tracking a moving mirror.

The weights are not arboOCR's work and are not covered by arboOCR's license,
which applies to arboOCR's own code only. The `NOTICE` file in the upstream
PaddleOCR repository is authoritative for the weights.
