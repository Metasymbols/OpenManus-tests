import asyncio
import io
import os
import tarfile
import tempfile
import uuid
from typing import Dict, Optional

import docker
from docker.errors import NotFound
from docker.models.containers import Container

from app.config import SandboxSettings
from app.sandbox.core.exceptions import SandboxTimeoutError
from app.sandbox.core.terminal import AsyncDockerizedTerminal


class DockerSandbox:
    """Docker沙箱环境。

    提供一个带有资源限制、文件操作和命令执行功能的容器化执行环境。

    Attributes:
        config: 沙箱配置。
        volume_bindings: 卷映射配置。
        client: Docker客户端。
        container: Docker容器实例。
        terminal: 容器终端接口。
    """

    def __init__(
        self,
        config: Optional[SandboxSettings] = None,
        volume_bindings: Optional[Dict[str, str]] = None,
    ):
        """初始化沙箱实例。

        Args:
            config: 沙箱配置。如果为None则使用默认配置。
            volume_bindings: 卷映射，格式为{主机路径: 容器路径}。
        """
        self.config = config or SandboxSettings()
        self.volume_bindings = volume_bindings or {}
        self.client = docker.from_env()
        self.container: Optional[Container] = None
        self.terminal: Optional[AsyncDockerizedTerminal] = None

    async def create(self) -> "DockerSandbox":
        """创建并启动沙箱容器。

        Returns:
            当前沙箱实例。

        Raises:
            docker.errors.APIError: 如果Docker API调用失败。
            RuntimeError: 如果容器创建或启动失败。
        """
        try:
            # Prepare container config
            host_config = self.client.api.create_host_config(
                mem_limit=self.config.memory_limit,
                cpu_period=100000,
                cpu_quota=int(100000 * self.config.cpu_limit),
                network_mode="none" if not self.config.network_enabled else "bridge",
                binds=self._prepare_volume_bindings(),
            )

            # Generate unique container name with sandbox_ prefix
            container_name = f"sandbox_{uuid.uuid4().hex[:8]}"

            # Create container
            container = await asyncio.to_thread(
                self.client.api.create_container,
                image=self.config.image,
                command="tail -f /dev/null",
                hostname="sandbox",
                working_dir=self.config.work_dir,
                host_config=host_config,
                name=container_name,
                tty=True,
                detach=True,
            )

            self.container = self.client.containers.get(container["Id"])

            # Start container
            await asyncio.to_thread(self.container.start)

            # Initialize terminal
            self.terminal = AsyncDockerizedTerminal(
                container["Id"],
                self.config.work_dir,
                env_vars={"PYTHONUNBUFFERED": "1"},
                # Ensure Python output is not buffered
            )
            await self.terminal.init()

            return self

        except Exception as e:
            await self.cleanup()  # Ensure resources are cleaned up
            raise RuntimeError(f"Failed to create sandbox: {e}") from e

    def _prepare_volume_bindings(self) -> Dict[str, Dict[str, str]]:
        """准备卷绑定配置。

        Returns:
            卷绑定配置字典。
        """
        bindings = {}

        # Create and add working directory mapping
        work_dir = self._ensure_host_dir(self.config.work_dir)
        bindings[work_dir] = {"bind": self.config.work_dir, "mode": "rw"}

        # Add custom volume bindings
        for host_path, container_path in self.volume_bindings.items():
            bindings[host_path] = {"bind": container_path, "mode": "rw"}

        return bindings

    @staticmethod
    def _ensure_host_dir(path: str) -> str:
        """确保主机上的目录存在。

        Args:
            path: 目录路径。

        Returns:
            主机上的实际路径。
        """
        host_path = os.path.join(
            tempfile.gettempdir(),
            f"sandbox_{os.path.basename(path)}_{os.urandom(4).hex()}",
        )
        os.makedirs(host_path, exist_ok=True)
        return host_path

    async def run_command(self, cmd: str, timeout: Optional[int] = None) -> str:
        """在沙箱中运行命令。

        Args:
            cmd: 要执行的命令。
            timeout: 超时时间（秒）。

        Returns:
            命令输出字符串。

        Raises:
            RuntimeError: 如果沙箱未初始化或命令执行失败。
            TimeoutError: 如果命令执行超时。
        """
        if not self.terminal:
            raise RuntimeError("Sandbox not initialized")

        try:
            return await self.terminal.run_command(
                cmd, timeout=timeout or self.config.timeout
            )
        except TimeoutError:
            raise SandboxTimeoutError(
                f"Command execution timed out after {timeout or self.config.timeout} seconds"
            )

    async def read_file(self, path: str) -> str:
        """从容器中读取文件。

        Args:
            path: 文件路径。

        Returns:
            文件内容字符串。

        Raises:
            FileNotFoundError: 如果文件不存在。
            RuntimeError: 如果读取操作失败。
        """
        if not self.container:
            raise RuntimeError("Sandbox not initialized")

        try:
            # Get file archive
            resolved_path = self._safe_resolve_path(path)
            tar_stream, _ = await asyncio.to_thread(
                self.container.get_archive, resolved_path
            )

            # Read file content from tar stream
            content = await self._read_from_tar(tar_stream)
            return content.decode("utf-8")

        except NotFound:
            raise FileNotFoundError(f"File not found: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

    async def write_file(self, path: str, content: str) -> None:
        """将内容写入容器中的文件。

        Args:
            path: Target path.
            content: File content.

        Raises:
            RuntimeError: If write operation fails.
        """
        if not self.container:
            raise RuntimeError("Sandbox not initialized")

        try:
            resolved_path = self._safe_resolve_path(path)
            parent_dir = os.path.dirname(resolved_path)

            # Create parent directory
            if parent_dir:
                await self.run_command(f"mkdir -p {parent_dir}")

            # Prepare file data
            tar_stream = await self._create_tar_stream(
                os.path.basename(path), content.encode("utf-8")
            )

            # Write file
            await asyncio.to_thread(
                self.container.put_archive, parent_dir or "/", tar_stream
            )

        except Exception as e:
            raise RuntimeError(f"Failed to write file: {e}")

    def _safe_resolve_path(self, path: str) -> str:
        """Safely resolves container path, preventing path traversal.

        Args:
            path: Original path.

        Returns:
            Resolved absolute path.

        Raises:
            ValueError: If path contains potentially unsafe patterns.
        """
        # Check for path traversal attempts
        if ".." in path.split("/"):
            raise ValueError("Path contains potentially unsafe patterns")

        resolved = (
            os.path.join(self.config.work_dir, path)
            if not os.path.isabs(path)
            else path
        )
        return resolved

    async def copy_from(self, src_path: str, dst_path: str) -> None:
        """Copies a file from the container.

        Args:
            src_path: Source file path (container).
            dst_path: Destination path (host).

        Raises:
            FileNotFoundError: If source file does not exist.
            RuntimeError: If copy operation fails.
        """
        try:
            # Ensure destination file's parent directory exists
            parent_dir = os.path.dirname(dst_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            # Get file stream
            resolved_src = self._safe_resolve_path(src_path)
            stream, stat = await asyncio.to_thread(
                self.container.get_archive, resolved_src
            )

            # Create temporary directory to extract file
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Write stream to temporary file
                tar_path = os.path.join(tmp_dir, "temp.tar")
                with open(tar_path, "wb") as f:
                    for chunk in stream:
                        f.write(chunk)

                # Extract file
                with tarfile.open(tar_path) as tar:
                    members = tar.getmembers()
                    if not members:
                        raise FileNotFoundError(f"Source file is empty: {src_path}")

                    # If destination is a directory, we should preserve relative path structure
                    if os.path.isdir(dst_path):
                        tar.extractall(dst_path)
                    else:
                        # If destination is a file, we only extract the source file's content
                        if len(members) > 1:
                            raise RuntimeError(
                                f"Source path is a directory but destination is a file: {src_path}"
                            )

                        with open(dst_path, "wb") as dst:
                            src_file = tar.extractfile(members[0])
                            if src_file is None:
                                raise RuntimeError(
                                    f"Failed to extract file: {src_path}"
                                )
                            dst.write(src_file.read())

        except docker.errors.NotFound:
            raise FileNotFoundError(f"Source file not found: {src_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to copy file: {e}")

    async def copy_to(self, src_path: str, dst_path: str) -> None:
        """Copies a file to the container.

        Args:
            src_path: Source file path (host).
            dst_path: Destination path (container).

        Raises:
            FileNotFoundError: If source file does not exist.
            RuntimeError: If copy operation fails.
        """
        try:
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"Source file not found: {src_path}")

            # Create destination directory in container
            resolved_dst = self._safe_resolve_path(dst_path)
            container_dir = os.path.dirname(resolved_dst)
            if container_dir:
                await self.run_command(f"mkdir -p {container_dir}")

            # Create tar file to upload
            with tempfile.TemporaryDirectory() as tmp_dir:
                tar_path = os.path.join(tmp_dir, "temp.tar")
                with tarfile.open(tar_path, "w") as tar:
                    # Handle directory source path
                    if os.path.isdir(src_path):
                        os.path.basename(src_path.rstrip("/"))
                        for root, _, files in os.walk(src_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.join(
                                    os.path.basename(dst_path),
                                    os.path.relpath(file_path, src_path),
                                )
                                tar.add(file_path, arcname=arcname)
                    else:
                        # Add single file to tar
                        tar.add(src_path, arcname=os.path.basename(dst_path))

                # Read tar file content
                with open(tar_path, "rb") as f:
                    data = f.read()

                # Upload to container
                await asyncio.to_thread(
                    self.container.put_archive,
                    os.path.dirname(resolved_dst) or "/",
                    data,
                )

                # Verify file was created successfully
                try:
                    await self.run_command(f"test -e {resolved_dst}")
                except Exception:
                    raise RuntimeError(f"Failed to verify file creation: {dst_path}")

        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to copy file: {e}")

    @staticmethod
    async def _create_tar_stream(name: str, content: bytes) -> io.BytesIO:
        """Creates a tar file stream.

        Args:
            name: Filename.
            content: File content.

        Returns:
            Tar file stream.
        """
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tarinfo = tarfile.TarInfo(name=name)
            tarinfo.size = len(content)
            tar.addfile(tarinfo, io.BytesIO(content))
        tar_stream.seek(0)
        return tar_stream

    @staticmethod
    async def _read_from_tar(tar_stream) -> bytes:
        """Reads file content from a tar stream.

        Args:
            tar_stream: Tar file stream.

        Returns:
            File content.

        Raises:
            RuntimeError: If read operation fails.
        """
        with tempfile.NamedTemporaryFile() as tmp:
            for chunk in tar_stream:
                tmp.write(chunk)
            tmp.seek(0)

            with tarfile.open(fileobj=tmp) as tar:
                member = tar.next()
                if not member:
                    raise RuntimeError("Empty tar archive")

                file_content = tar.extractfile(member)
                if not file_content:
                    raise RuntimeError("Failed to extract file content")

                return file_content.read()

    async def cleanup(self) -> None:
        """Cleans up sandbox resources."""
        errors = []
        try:
            if self.terminal:
                try:
                    await self.terminal.close()
                except Exception as e:
                    errors.append(f"Terminal cleanup error: {e}")
                finally:
                    self.terminal = None

            if self.container:
                try:
                    await asyncio.to_thread(self.container.stop, timeout=5)
                except Exception as e:
                    errors.append(f"Container stop error: {e}")

                try:
                    await asyncio.to_thread(self.container.remove, force=True)
                except Exception as e:
                    errors.append(f"Container remove error: {e}")
                finally:
                    self.container = None

        except Exception as e:
            errors.append(f"General cleanup error: {e}")

        if errors:
            print(f"Warning: Errors during cleanup: {', '.join(errors)}")

    async def __aenter__(self) -> "DockerSandbox":
        """Async context manager entry."""
        return await self.create()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.cleanup()
