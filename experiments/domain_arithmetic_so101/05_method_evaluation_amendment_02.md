# Amendment 02 method evaluation

Status: implementation is method-compatible, with one environment requirement corrected in the
README before release.

The implementation preserves the DArT comparison: source and target updates are independently
fine-tuned from the same pinned SmolVLA base, then direct arithmetic and DArT are merged with frozen
alpha, rank, seed, and tensor-key validation. The replacement target is one same-rig episode and is
read directly from its immutable Hub revision without task, frame, camera, or gripper conversion.
The target contract, pinned episode, loader settings, source-checkpoint override, RTC runtime, and
merge provenance have focused tests or shell checks.

The targeted pytest collection could not run in this WSL environment because the installed PyTorch
could not load `libcublasLt.so.12`; this is an environment blocker, not a reported test failure.
Run the tests on the remote GPU after installing the documented extras. The required clean-server
setup is `uv sync --locked --extra training --extra smolvla --extra feetech`; the policy and hardware
extras alone do not install datasets, PyAV, or training dependencies.

No formal rollout result is supported by this implementation review. Prior hardware smoke tests
remain pilot-only under Amendment 02, and source-checkpoint reuse still requires verifying the
frozen base/configuration, processor hashes, final step, and model hash.
