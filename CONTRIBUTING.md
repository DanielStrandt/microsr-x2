# Contributing

Thank you for helping improve MicroSR-X2.

1. Open an issue describing the bug or enhancement; include Python, PyTorch, OS and CPU/GPU information.
2. Use a virtual environment and run `python -m pip install -r requirements-dev.txt`.
3. Add targeted tests for model-format, tiling, output-color or file handling changes; run `python -m pytest -q`.
4. Keep the **396 INT2-code** model/storage claims accurate. Changes to weights, scales, biases, preprocessing or crop protocols can affect benchmarks and must be disclosed.
5. Never attach copyrighted/private training images, private `.pt` checkpoints, or sensitive paths in issues.
6. Cite full inference file size and PyTorch dependency rather than implying the runtime is 495 bytes.

Pull requests should explain the change, affected outputs, and how you verified parity against previous behavior. No contributor license agreement is included.
