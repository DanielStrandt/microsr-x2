# Security policy

For suspected security vulnerabilities, **do not post working exploit details in a public issue**. Until a dedicated maintainer contact is configured, use GitHub's private vulnerability reporting feature if enabled for the repository, or contact the repository owner privately.

## Input trust

- Public inference loads a validated compact binary `.msr`, or JSON/packed-code export, and **does not use Python pickle/`torch.load`**.
- Optional **research-only** checkpoint loading uses PyTorch serialization. Do **not** load `.pt` or other serialized checkpoints from untrusted sources.
- The `.msr` file checks a CRC32 for accidental corruption; CRC32 **does not authenticate a model**. Use signed releases or a trusted SHA-256 hash where security matters.
- Image decoders (Pillow) and PyTorch are third-party dependencies. Keep them updated, and do not run image-processing tools with unnecessary privileges.
- Very large images may exhaust RAM or VRAM. Use tiling and system-level input limits for untrusted uploads; the desktop GUI is not a hardened multi-user web service.

Security patches should preserve backward-compatible model reading when safe, and introduce a new model format version when necessary.
