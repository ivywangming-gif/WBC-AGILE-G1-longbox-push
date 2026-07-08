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
- WebRTC/livestream in WSL currently black-screen due CUDA/Vulkan interop/shared resource failure
- Prioritize headless numerical validation and audit scripts before GUI

Pending:
- Debug-G1-Object-v0 train smoke test
- G1-PickPlace-Tracking-v0 train smoke test
