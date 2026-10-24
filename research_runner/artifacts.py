"""Pure filesystem artifacts, compatibility signatures, and verified ZIP backups.

The parent runner owns prompts and execution policy. This module never asks for
confirmation, imports model libraries, reads dotenv files, or resolves secrets.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import traceback
from typing import Any, Iterable, Iterator
import uuid
import zipfile

from .config import is_secret_key
from .progress import emit, iterable


class ArtifactError(ValueError):
    """An unsafe, incomplete, incompatible, or damaged artifact."""


class ExistingResultsError(FileExistsError):
    """Existing results require an explicit overwrite or compatible resume."""


class IncompatibleResultsError(ArtifactError):
    """A completed result does not match the requested signature."""


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":"), default=_json_default).encode("utf-8")


def fingerprint(value: Any) -> str:
    """SHA-256 of canonical JSON; independent of dictionary order and environment."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    path = Path(path)
    size = path.stat().st_size
    large = size >= 8 * 1024 * 1024
    count = 0
    if large:
        emit('hash.file', 0, size, unit='bytes', message=str(path), force=True)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
            if large:
                emit('hash.file', count, size, unit='bytes', message=str(path))
    if large:
        emit('hash.file', count, size, unit='bytes', message=str(path), status='completed', force=True)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_bytes(path: Path, chunks: Iterable[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for chunk in chunks:
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: str | Path, value: Any) -> None:
    _atomic_bytes(Path(path), [_json_bytes(value) + b"\n"])


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    _atomic_bytes(Path(path), (_json_bytes(row) + b"\n" for row in rows))


def read_jsonl(path: str | Path) -> list[Any]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ArtifactError(f"Invalid JSONL at {path}:{number}") from exc
    return rows


def _public_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: item if key == "schema" else _public_config(item) for key, item in value.items()
                if not is_secret_key(str(key))}
    if isinstance(value, (list, tuple)):
        return [_public_config(item) for item in value]
    return value


def _error_text(value: str) -> str:
    value = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [REDACTED]", value)
    return re.sub(r"(?i)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret|authorization)"
                  r"[\"']?\s*[:=]\s*)(?:[\"'][^\"'\n]*[\"']|[^\s,;}]+)",
                  r"\1[REDACTED]", value)


def _source_commit(config: dict) -> str | None:
    """Read Git HEAD without launching a process (also supports linked worktrees)."""
    directory = Path(config.get("paths", {}).get("workspace", ".")) / ".git"
    try:
        if directory.is_file():
            pointer = directory.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir: "):
                return None
            directory = (directory.parent / pointer[8:]).resolve()
        common = directory
        if (directory / "commondir").is_file():
            common = (directory / (directory / "commondir").read_text(encoding="utf-8").strip()).resolve()
        commit = (directory / "HEAD").read_text(encoding="utf-8").strip()
        if commit.startswith("ref: "):
            reference = commit[5:]
            _safe_relative(reference)
            if not reference.startswith("refs/"):
                return None
            for parent in (directory, common):
                if (parent / reference).is_file():
                    commit = (parent / reference).read_text(encoding="utf-8").strip()
                    break
            else:
                commit = ""
                with (common / "packed-refs").open(encoding="utf-8") as handle:
                    for line in handle:
                        fields = line.strip().split()
                        if len(fields) == 2 and fields[1] == reference:
                            commit = fields[0]
                            break
    except (OSError, ValueError):
        return None
    return commit if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) else None


def experiment_signature(config: dict, name: str, inputs: Any = None) -> str:
    """Fingerprint scientific settings for one experiment, not execution controls.

    Changing another experiment, output locations, stage selection, or resume
    does not invalidate this experiment. Pass input/checkpoint content hashes in
    ``inputs`` to bind a run to exact source bytes (paths alone are insufficient).
    """
    name = name.split("/", 1)[0]
    settings = config.get("experiments", {}).get(name)
    if settings is None:
        raise KeyError(f"Unknown experiment: {name}")
    scientific = {key: config[key] for key in ("version", "data", "training", "models", "runtime")
                  if key in config}
    scientific["experiment"] = {key: value for key, value in settings.items()
                                if key not in {"stage", "resume", "overwrite"}}
    return fingerprint({"name": name, "config": _public_config(scientific), "inputs": inputs})


def _safe_relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ArtifactError(f"Unsafe relative path: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or ":" in part for part in parts):
        raise ArtifactError(f"Unsafe relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ArtifactError(f"Unsafe relative path: {value!r}")
    return path


def _files(directory: Path) -> dict[str, str]:
    result = {}
    emit('artifact.hashes', 0, message=f'Hashing/validating files: {directory}', unit='files', force=True)
    for parent, directories, files in os.walk(directory, followlinks=False):
        for name in sorted(directories + files):
            path = Path(parent) / name
            relative = path.relative_to(directory).as_posix()
            _safe_relative(relative)
            if path.is_symlink():
                raise ArtifactError(f"Artifact symlinks are not allowed: {relative}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ArtifactError(f"Artifact is not a regular file: {relative}")
            if relative != "manifest.json":
                result[relative] = file_sha256(path)
                emit('artifact.hashes', len(result), message=f'{directory.name}: {relative}', unit='files')
    emit('artifact.hashes', len(result), unit='files', message=str(directory), status='completed', force=True)
    return result


class ArtifactStore:
    """Transactional result directories rooted at ``root``.

    New/overwrite transactions yield writable staging Paths. Compatible completed
    resume yields the existing verified Path for read-only use. Stages must have
    separate names: resuming Stage 1 never skips Stage 2. ``load`` is also a
    verification/read API; callers must not mutate completed paths.
    """

    def __init__(self, root: str | Path, config: dict):
        self.root = Path(root).expanduser().resolve()
        self.config = config

    def signature(self, name: str, inputs: Any = None) -> str:
        return experiment_signature(self.config, name, inputs)

    def _target(self, name: str) -> Path:
        relative = _safe_relative(name)
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) for part in relative.parts):
            raise ArtifactError(f"Unsafe experiment name: {name!r}")
        target = self.root.joinpath(*relative.parts)
        for component in (target, *target.parents):
            if component == self.root:
                break
            if component.is_symlink():
                raise ArtifactError(f"Artifact paths cannot contain symlinks: {component}")
        return target

    def _internal_directory(self, name: str) -> Path:
        directory = self.root / name
        if directory.is_symlink():
            raise ArtifactError(f"Internal artifact directory cannot be a symlink: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _retain_failure(self, stage: Path, name: str, signature: str, error: BaseException) -> None:
        """Best-effort diagnostics only; incomplete model/data artifacts are discarded."""
        if stage.is_symlink() or not stage.is_dir():
            return
        failed = self._internal_directory(".failed")
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=failed))
        try:
            for parent, directories, files in os.walk(stage, followlinks=False):
                parent = Path(parent)
                directories[:] = [entry for entry in directories if not (parent / entry).is_symlink()
                                  and not _excluded(PurePosixPath((parent / entry).relative_to(stage).as_posix()), [])]
                for entry in files:
                    source = parent / entry
                    relative = source.relative_to(stage).as_posix()
                    if (source.is_symlink() or not source.is_file()
                            or _excluded(_safe_relative(relative), [])
                            or not (source.suffix.lower() == ".log"
                                    or source.name in {"stderr.txt", "stdout.txt", "error.txt", "errors.txt"})):
                        continue
                    with source.open(encoding="utf-8", errors="replace") as handle:
                        _atomic_bytes(temporary / "logs" / relative,
                                      (_error_text(line).encode("utf-8") for line in handle))
            write_json(temporary / "failure.json", {
                "exception_type": type(error).__name__, "message": _error_text(str(error)),
                "traceback": _error_text("".join(traceback.format_exception(type(error), error, error.__traceback__))),
                "resume_policy": "failed stages are rerun; only completed artifacts may resume",
            })
            write_json(temporary / "effective_config.json", _public_config(self.config))
            write_json(temporary / "manifest.json", {
                "version": 1, "name": name, "signature": signature, "status": "failed",
                "files": _files(temporary), "source_commit": _source_commit(self.config),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            os.replace(temporary, failed / f"{name.replace('/', '-')}_{stamp}_{uuid.uuid4().hex[:8]}")
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def load(self, name: str, signature: str | None = None) -> Path:
        """Return completed results only after signature and all file hashes match."""
        target = self._target(name)
        if not target.exists():
            raise FileNotFoundError(target)
        try:
            if not target.is_dir() or (target / "manifest.json").is_symlink():
                raise ArtifactError(f"Unsafe result directory: {target}")
            manifest = read_json(target / "manifest.json")
            if not isinstance(manifest, dict) or type(manifest.get("version")) is not int or manifest["version"] != 1:
                raise ArtifactError(f"Invalid manifest: {target}")
            if manifest.get("status") != "completed" or manifest.get("name") != name:
                raise ArtifactError(f"Results are not completed: {target}")
            stored_signature = manifest.get("signature")
            if not isinstance(stored_signature, str) or not stored_signature:
                raise ArtifactError(f"Missing compatibility signature: {target}")
            if signature is not None and stored_signature != signature:
                raise IncompatibleResultsError(f"Incompatible results: {name}")
            expected = manifest.get("files")
            if not isinstance(expected, dict):
                raise ArtifactError(f"Missing file hash manifest: {target}")
            for relative, digest in expected.items():
                _safe_relative(relative)
                if relative == "manifest.json" or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ArtifactError(f"Invalid file hash manifest: {target}")
            if _files(target) != expected:
                raise ArtifactError(f"Artifact file hashes do not match: {target}")
        except (OSError, ValueError, TypeError) as exc:
            if isinstance(exc, ArtifactError):
                raise
            raise ArtifactError(f"Cannot verify artifacts: {target}") from exc
        return target

    def ensure(self, name: str, signature: str, *, overwrite: bool = False,
               resume: bool = False) -> Path | None:
        """Nonmutating preflight; parent can aggregate overwrite confirmation once."""
        if overwrite and resume:
            raise ValueError("overwrite and resume are mutually exclusive")
        target = self._target(name)
        if not target.exists():
            return None
        if resume:
            return self.load(name, signature)
        if not overwrite:
            raise ExistingResultsError(f"Results already exist: {target}; overwrite defaults to No")
        return None

    @contextmanager
    def transaction(self, name: str, signature: str, overwrite: bool = False,
                    resume: bool = False) -> Iterator[Path]:
        """Stage work and publish a completed manifest, rolling back on failure."""
        if not isinstance(signature, str) or not signature:
            raise ValueError("signature must be a nonempty string")
        self.ensure(name, signature, overwrite=overwrite, resume=resume)
        target = self._target(name)
        lock_root = self._internal_directory(".locks")
        lock = lock_root / (fingerprint(name) + ".lock")
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise ExistingResultsError(f"An artifact transaction is already active: {name}") from exc
        os.close(descriptor)
        stage = None
        previous = None
        try:
            completed = self.ensure(name, signature, overwrite=overwrite, resume=resume)
            if completed is not None:
                emit('artifact.resume', message=f'Using verified completed artifact: {name}', force=True)
                yield completed
                self.load(name, signature)
                return
            staging = self._internal_directory(".staging")
            stage = Path(tempfile.mkdtemp(prefix="run-", dir=staging))
            yield stage
            if stage.is_symlink() or not stage.is_dir():
                raise ArtifactError("Staging directory was replaced or removed")
            write_json(stage / "effective_config.json", _public_config(self.config))
            emit('artifact.publish', message=f'Verifying and atomically publishing {name}.', force=True)
            files = _files(stage)
            if (stage / "manifest.json").exists():
                raise ArtifactError("manifest.json is reserved for the artifact store")
            write_json(stage / "manifest.json", {
                "version": 1, "name": name, "signature": signature,
                "status": "completed", "files": files,
                "config_fingerprint": fingerprint(_public_config(self.config)),
                "source_commit": _source_commit(self.config),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            # Recheck paths after arbitrary caller work; never publish through a link.
            target = self._target(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not (overwrite or resume):
                    raise ExistingResultsError(f"Results appeared during transaction: {target}")
                previous = target.parent / f".{target.name}.previous-{uuid.uuid4().hex}"
                os.replace(target, previous)
            try:
                os.replace(stage, target)
            except BaseException:
                if previous is not None:
                    os.replace(previous, target)
                    previous = None
                raise
            stage = None
            emit('artifact.publish', status='completed', message=f'Published {name}.', force=True)
            if previous is not None:
                # Publication succeeded; cleanup must not turn success into failure.
                if previous.is_dir():
                    shutil.rmtree(previous, ignore_errors=True)
                else:
                    previous.unlink(missing_ok=True)
                previous = None
        except BaseException as error:
            if stage is not None:
                try:
                    self._retain_failure(stage, name, signature, error)
                except Exception:
                    # Diagnostics must never replace the original worker/publication error.
                    pass
            raise
        finally:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)
            lock.unlink(missing_ok=True)

    def backup(self) -> Path | None:
        return create_backup(self.config, results_root=self.root)


_EXCLUDED_PARTS = {".git", ".env", ".aws", ".ssh", ".cache", "cache", "caches",
                   "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                   ".venv", "venv", "env", "envs", "node_modules", "secrets",
                   "secret", "credentials", ".staging", ".locks", ".progress"}


def _excluded(relative: PurePosixPath, patterns: list[str]) -> bool:
    lowered = [part.lower() for part in relative.parts]
    return (any(part in _EXCLUDED_PARTS or part.startswith(".env") or ".previous-" in part
                for part in lowered)
            or relative.suffix.lower() in {".zip", ".pyc", ".pyo", ".pem", ".key"}
            or relative.name.lower() in {".netrc", ".npmrc", "id_rsa", "id_ed25519"}
            or is_secret_key(relative.stem)
            or any(part.startswith(("secret.", "secrets.", "credentials.")) for part in lowered)
            or any(fnmatch.fnmatchcase(relative.as_posix(), pattern) for pattern in patterns))


def _checkpoint_sources(config: dict, workspace: Path) -> list[tuple[str, Path]]:
    sources = []
    for name, value in config.get("models", {}).items():
        if isinstance(value, str) and name in {"encoder", "autoencoder"}:
            candidates = [(name, value)]
        elif isinstance(value, dict):
            candidates = [(f"{name}/{key}", item) for key, item in value.items()
                          if key in {"path", "checkpoint", "checkpoint_path", "checkpoint_dir",
                                     "encoder_path", "autoencoder_path", "ae_path", "model_path"}
                          and isinstance(item, str)]
        else:
            candidates = []
        for alias, candidate in candidates:
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = workspace / path
            if path.exists():
                sources.append((f"checkpoints/{alias}", path))
    for index, value in enumerate(config.get("backup", {}).get("checkpoint_paths", [])):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = workspace / path
        if not path.exists():
            raise FileNotFoundError(f"Configured checkpoint backup source is missing: {path}")
        sources.append((f"checkpoints/extra-{index}", path))
    return sources


def create_backup(config: dict, *, results_root: str | Path | None = None) -> Path | None:
    """Build a filtered ZIP, then verify CRC and SHA-256 before publishing it.

    Paths in the archive are aliases under results/ and checkpoints/, even when
    sources live outside the workspace. All symlinks are excluded, including
    in-root links; no archives, caches, dotenv, secrets, or git data are nested.
    Explicitly enabled checkpoint formats may themselves use ZIP serialization.
    """
    options = config.get("backup", {})
    if not options.get("enabled", True):
        return None
    paths = config.get("paths", {})
    workspace = Path(paths.get("workspace", ".")).expanduser().resolve()
    output = Path(paths.get("backup", "backup")).expanduser()
    if not output.is_absolute():
        output = workspace / output
    output = output.resolve()
    results = Path(results_root if results_root is not None else paths.get("results", "results/research_runner"))
    if not results.is_absolute():
        results = workspace / results
    sources = []
    if options.get("include_results", True) and results.exists():
        sources.append(("results", results))
    if options.get("include_checkpoints", False):
        sources.extend(_checkpoint_sources(config, workspace))
    for index, value in enumerate(options.get("include_paths", [])):
        source = Path(value).expanduser()
        if not source.is_absolute():
            source = workspace / source
        if not source.exists():
            raise FileNotFoundError(f"Configured backup source is missing: {source}")
        sources.append((f"additional/{index}-{source.name}", source))
    patterns = options.get("exclude", []) + options.get("exclude_patterns", [])
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = output / f"research_{timestamp}_{uuid.uuid4().hex[:8]}.zip"
    descriptor, temporary = tempfile.mkstemp(prefix=".backup-", suffix=".tmp", dir=output)
    os.close(descriptor)
    temporary = Path(temporary)
    checksums = {}
    emit('backup.scan', message='Scanning configured sources; excluding live progress, secrets and nested ZIPs.', force=True)
    emit('backup.files', 0, unit='files', force=True)

    def add(archive: zipfile.ZipFile, source: Path, arcname: str) -> None:
        relative = _safe_relative(arcname)
        source_parts = PurePosixPath(*source.parts[1:]) if source.is_absolute() else PurePosixPath(source.as_posix())
        if (_excluded(relative, patterns) or _excluded(source_parts, [])
                or source.is_symlink() or not source.is_file()):
            return
        include_checkpoints = options.get("include_checkpoints", False)
        checkpoint_suffix = source.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin"}
        if not include_checkpoints and (checkpoint_suffix or source.suffix.lower() == ".safetensors"):
            return
        if source.resolve().is_relative_to(output):
            return
        # torch.save uses ZIP containers; these are weights, not nested backups.
        if zipfile.is_zipfile(source) and not (include_checkpoints and checkpoint_suffix):
            return
        if arcname in checksums:
            raise ArtifactError(f"Duplicate archive path: {arcname}")
        digest = hashlib.sha256()
        size, copied = source.stat().st_size, 0
        large = size >= 8 * 1024 * 1024
        if large:
            emit('backup.copy', 0, size, unit='bytes', message=arcname, force=True)
        with source.open("rb") as handle, archive.open(arcname, "w", force_zip64=True) as member:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                member.write(chunk)
                copied += len(chunk)
                if large:
                    emit('backup.copy', copied, size, unit='bytes', message=arcname)
        if large:
            emit('backup.copy', copied, size, unit='bytes', message=arcname, status='completed', force=True)
        checksums[arcname] = digest.hexdigest()
        emit('backup.files', len(checksums), unit='files', message=arcname)

    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for alias, source in sources:
                _safe_relative(alias)
                if source.is_symlink():
                    continue
                if source.is_file():
                    add(archive, source, f"{alias}/{source.name}")
                    continue
                for parent, directories, files in os.walk(source, followlinks=False):
                    parent = Path(parent)
                    directories[:] = sorted(name for name in directories
                                            if not (parent / name).is_symlink()
                                            and not _excluded(PurePosixPath(alias) / (parent / name).relative_to(source).as_posix(), patterns)
                                            and not (parent / name).resolve().is_relative_to(output))
                    for name in sorted(files):
                        file = parent / name
                        add(archive, file, f"{alias}/{file.relative_to(source).as_posix()}")
            if options.get("include_effective_config", True):
                effective = _json_bytes(_public_config(config)) + b"\n"
                archive.writestr("effective_config.json", effective)
                checksums["effective_config.json"] = hashlib.sha256(effective).hexdigest()
            if config.get("_config_path"):
                source = Path(config["_config_path"])
                expected = config.get("_config_sha256")
                if expected is None or source.is_symlink() or file_sha256(source) != expected:
                    raise ArtifactError("Source YAML changed after configuration loading")
                add(archive, source, "source_config.yaml")
                if checksums.get("source_config.yaml") != expected:
                    raise ArtifactError("Source YAML could not be safely archived")
            archive.writestr("checksums.json", _json_bytes({"version": 1, "files": checksums}) + b"\n")
        with zipfile.ZipFile(temporary) as archive:
            emit('backup.crc', message='Checking ZIP CRC integrity.', force=True)
            if archive.testzip() is not None:
                raise ArtifactError("Backup CRC verification failed")
            manifest = json.loads(archive.read("checksums.json"))
            if manifest != {"version": 1, "files": checksums}:
                raise ArtifactError("Backup checksum manifest differs")
            if set(archive.namelist()) != set(checksums) | {"checksums.json"}:
                raise ArtifactError("Backup has unmanifested members")
            for name, expected in iterable(checksums.items(), 'backup.verify', unit='files'):
                digest = hashlib.sha256()
                size, verified = archive.getinfo(name).file_size, 0
                large = size >= 8 * 1024 * 1024
                if large:
                    emit('backup.verify_bytes', 0, size, unit='bytes', message=name, force=True)
                with archive.open(name) as member:
                    for chunk in iter(lambda: member.read(1024 * 1024), b""):
                        digest.update(chunk)
                        verified += len(chunk)
                        if large:
                            emit('backup.verify_bytes', verified, size, unit='bytes', message=name)
                if digest.hexdigest() != expected:
                    raise ArtifactError(f"Backup checksum mismatch: {name}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        emit('backup.publish', status='completed', message=str(destination), force=True)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def backup_snapshot(config: dict) -> Path | None:
    """Parent-runner entry point for the configured verified backup."""
    return create_backup(config)
