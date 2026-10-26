"""Process-local build tools for the opt-in modern Audit runtime only."""
import ast
from importlib import metadata
import os
from pathlib import Path
import re
import shutil
import subprocess


class PreflightError(RuntimeError):
    """A public, credential-free dependency diagnostic."""


def worker_environment(python, backend, base=None):
    environment = dict(os.environ if base is None else base)
    if backend.get('structured_output_api') != 'structured_outputs':
        return environment
    binary = Path(python).expanduser().absolute().parent
    toolkit = binary.parent / 'ig_cuda_toolkit'
    environment.update(CUDA_HOME=str(toolkit), CUDA_PATH=str(toolkit),
                       CUDACXX=str(toolkit / 'bin/nvcc'), FLASHINFER_NVCC=str(toolkit / 'bin/nvcc'))
    environment['PATH'] = os.pathsep.join((str(binary), str(toolkit / 'bin'), environment.get('PATH', '')))
    return environment


def _version(command, environment):
    try:
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError(f'Runtime executable failed: {command[0]}. Repair the isolated Audit runtime.') from exc
    return result.stdout.strip()


def inspect_toolchain(python, backend, environment=None):
    environment = worker_environment(python, backend, environment)
    if backend.get('structured_output_api') != 'structured_outputs':
        return {}
    tools = {name: shutil.which(name, path=environment.get('PATH')) for name in ('ninja', 'nvcc', 'c++')}
    if any(not value for value in tools.values()):
        missing = ', '.join(name for name, value in tools.items() if not value)
        raise PreflightError(f'Missing worker build tools: {missing}. Run python -m research_runner.backend_setup --repair-runtime')
    toolkit = Path(environment['CUDA_HOME'])
    if not (toolkit / 'include/cuda_runtime.h').is_file() or not (toolkit / 'lib64/libcudart.so').is_file():
        raise PreflightError('Isolated CUDA compiler/linker layout is missing. Run python -m research_runner.backend_setup --repair-runtime')
    text = _version([tools['nvcc'], '--version'], environment)
    match = re.search(r'release (\d+\.\d+)', text)
    # Read build metadata without importing torch or initializing CUDA.
    tree = ast.parse(Path(metadata.distribution('torch').locate_file('torch/version.py')).read_text())
    cuda = next((ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'cuda'), None)
    if not match or match[1] != cuda:
        raise PreflightError(f'Worker nvcc CUDA {match[1] if match else "unknown"} does not match PyTorch CUDA {cuda}. Repair the isolated runtime.')
    return {'ninja': tools['ninja'], 'ninja_version': _version([tools['ninja'], '--version'], environment),
            'nvcc': tools['nvcc'], 'nvcc_cuda': match[1], 'torch_cuda': cuda,
            'cuda_home': str(toolkit), 'cxx': tools['c++']}


def create_toolkit_layout(python):
    """Create a venv-owned SDK view; never edit /usr or the reference venv."""
    root = Path(python).absolute().parent.parent
    matches = list(root.glob('lib/python*/site-packages/nvidia/cu13'))
    if len(matches) != 1 or not (matches[0] / 'bin/nvcc').is_file():
        raise PreflightError('Pinned NVIDIA CUDA compiler package is unavailable in the isolated environment')
    source, target = matches[0], root / 'ig_cuda_toolkit'
    if target.is_symlink():
        raise PreflightError('Refusing to overwrite a symlink at the isolated toolkit root')
    target.mkdir(exist_ok=True)

    def link(name, destination):
        if name.is_symlink():
            if name.resolve() == destination.resolve():
                return
            raise PreflightError(f'Existing toolkit link points elsewhere: {name}')
        if name.exists():
            raise PreflightError(f'Refusing to overwrite an existing toolkit file: {name}')
        name.symlink_to(destination)

    for name in ('bin', 'nvvm'):
        link(target / name, source / name)
    for name, directory in (('include', source / 'include'), ('lib64', source / 'lib')):
        output = target / name
        if output.is_symlink():
            raise PreflightError('Toolkit include/library directories must be local, not symlinks')
        output.mkdir(exist_ok=True)
        for item in directory.iterdir():
            link(output / item.name, item)
    if not (target / 'include/cccl').exists():
        link(target / 'include/cccl', source / 'cccl/include')
    link(target / 'lib64/libcudart.so', source / 'lib/libcudart.so.13')
    link(target / 'lib', target / 'lib64')
    return str(target)
