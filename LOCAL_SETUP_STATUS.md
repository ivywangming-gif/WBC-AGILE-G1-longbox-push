# Local WBC-AGILE setup status

Host:
- Windows + WSL2 Ubuntu 22.04.5
- Conda env: agile, Python 3.11
- IsaacLab: ~/IsaacLab, v2.3.2
- Isaac Sim pip: 5.1.0
- GPU visible through torch CUDA

Validated:
- scripts/verify_rsl_rl.py passes
- Velocity-G1-History-v0 headless eval reaches Running evaluation and exits with --num_steps
- Debug-G1-Object-v0 headless train smoke test passes
- G1-PickPlace-Tracking-v0 headless train smoke test passes

Known limitation:
- WSL WebRTC/livestream is black-screen due CUDA/Vulkan interop/shared-resource failure.
- For now, prioritize headless numerical validation and audit logs.
