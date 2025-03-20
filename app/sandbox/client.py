from abc import ABC, abstractmethod
from typing import Dict, Optional, Protocol

from app.config import SandboxSettings
from app.sandbox.core.sandbox import DockerSandbox


class SandboxFileOperations(Protocol):
    """沙箱文件操作协议。

    定义了沙箱容器内文件操作的基本接口。
    """

    async def copy_from(self, container_path: str, local_path: str) -> None:
        """从容器内复制文件到本地。

        Args:
            container_path: 容器内的文件路径。
            local_path: 本地目标路径。
        """
        ...

    async def copy_to(self, local_path: str, container_path: str) -> None:
        """从本地复制文件到容器。

        Args:
            local_path: 本地源文件路径。
            container_path: 容器内的目标路径。
        """
        ...

    async def read_file(self, path: str) -> str:
        """读取容器内文件内容。

        Args:
            path: 容器内的文件路径。

        Returns:
            str: 文件内容。
        """
        ...

    async def write_file(self, path: str, content: str) -> None:
        """写入内容到容器内的文件。

        Args:
            path: 容器内的文件路径。
            content: 要写入的内容。
        """
        ...


class BaseSandboxClient(ABC):
    """基础沙箱客户端接口。

    定义了与沙箱容器交互的基本操作接口。
    """

    @abstractmethod
    async def create(
        self,
        config: Optional[SandboxSettings] = None,
        volume_bindings: Optional[Dict[str, str]] = None,
    ) -> None:
        """创建沙箱容器。

        Args:
            config: 沙箱配置。
            volume_bindings: 卷映射配置。
        """

    @abstractmethod
    async def run_command(self, command: str, timeout: Optional[int] = None) -> str:
        """执行命令。

        Args:
            command: 要执行的命令。
            timeout: 超时时间（秒）。

        Returns:
            str: 命令执行输出。
        """

    @abstractmethod
    async def copy_from(self, container_path: str, local_path: str) -> None:
        """从容器复制文件。

        Args:
            container_path: 容器内的文件路径。
            local_path: 本地目标路径。
        """

    @abstractmethod
    async def copy_to(self, local_path: str, container_path: str) -> None:
        """复制文件到容器。

        Args:
            local_path: 本地源文件路径。
            container_path: 容器内的目标路径。
        """

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """读取文件。

        Args:
            path: 容器内的文件路径。

        Returns:
            str: 文件内容。
        """

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """写入文件。

        Args:
            path: 容器内的文件路径。
            content: 要写入的内容。
        """

    @abstractmethod
    async def cleanup(self) -> None:
        """清理资源。

        清理沙箱容器及相关资源。
        """


class LocalSandboxClient(BaseSandboxClient):
    """本地沙箱客户端实现。

    实现了基础沙箱客户端接口，提供本地Docker容器的操作功能。
    """

    def __init__(self):
        """初始化本地沙箱客户端。"""
        self.sandbox: Optional[DockerSandbox] = None

    async def create(
        self,
        config: Optional[SandboxSettings] = None,
        volume_bindings: Optional[Dict[str, str]] = None,
    ) -> None:
        """Creates a sandbox.

        Args:
            config: Sandbox configuration.
            volume_bindings: Volume mappings.

        Raises:
            RuntimeError: If sandbox creation fails.
        """
        self.sandbox = DockerSandbox(config, volume_bindings)
        await self.sandbox.create()

    async def run_command(self, command: str, timeout: Optional[int] = None) -> str:
        """Runs command in sandbox.

        Args:
            command: Command to execute.
            timeout: Execution timeout in seconds.

        Returns:
            Command output.

        Raises:
            RuntimeError: If sandbox not initialized.
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not initialized")
        return await self.sandbox.run_command(command, timeout)

    async def copy_from(self, container_path: str, local_path: str) -> None:
        """Copies file from container to local.

        Args:
            container_path: File path in container.
            local_path: Local destination path.

        Raises:
            RuntimeError: If sandbox not initialized.
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not initialized")
        await self.sandbox.copy_from(container_path, local_path)

    async def copy_to(self, local_path: str, container_path: str) -> None:
        """Copies file from local to container.

        Args:
            local_path: Local source file path.
            container_path: Destination path in container.

        Raises:
            RuntimeError: If sandbox not initialized.
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not initialized")
        await self.sandbox.copy_to(local_path, container_path)

    async def read_file(self, path: str) -> str:
        """Reads file from container.

        Args:
            path: File path in container.

        Returns:
            File content.

        Raises:
            RuntimeError: If sandbox not initialized.
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not initialized")
        return await self.sandbox.read_file(path)

    async def write_file(self, path: str, content: str) -> None:
        """Writes file to container.

        Args:
            path: File path in container.
            content: File content.

        Raises:
            RuntimeError: If sandbox not initialized.
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not initialized")
        await self.sandbox.write_file(path, content)

    async def cleanup(self) -> None:
        """清理资源。

        清理沙箱容器及相关资源。
        """
        if self.sandbox:
            await self.sandbox.cleanup()
            self.sandbox = None


def create_sandbox_client() -> LocalSandboxClient:
    """创建沙箱客户端。

    Returns:
        LocalSandboxClient: 沙箱客户端实例。
    """
    return LocalSandboxClient()


SANDBOX_CLIENT = create_sandbox_client()
