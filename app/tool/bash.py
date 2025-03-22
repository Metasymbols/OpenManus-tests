import asyncio
import os
from typing import Optional

from app.exceptions import ToolError
from app.tool.base import BaseTool, CLIResult


_BASH_DESCRIPTION = """在终端执行bash命令的交互式工具

特性说明：
1. 长时命令处理：
   - 需在后台运行的命令应重定向输出到文件（示例：`python3 app.py > server.log 2>&1 &`）
2. 交互控制：
   - 返回码-1表示进程未结束，可通过以下方式交互：
     • 发送空命令获取最新日志
     • 发送文本到进程标准输入
     • 发送`ctrl+c`中断进程
3. 超时机制：
   - 超时后自动发送SIGINT信号
   - 建议重试后台运行模式

与terminal.py的主要差异：
• 支持持续交互式会话
• 内置进程生命周期管理
• 提供异步超时控制机制
"""


class _BashSession:
    """Bash会话管理器（异步实现）

    特性：
    - 维持长期bash进程
    - 通过哨兵检测输出结束
    - 自动清理输出缓冲区

    生命周期：
    1. start() 初始化子进程
    2. run() 执行多个命令
    3. stop() 终止进程

    超时控制：
    - 默认120秒执行超时
    - 检测到超时自动标记会话状态
    """

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """Terminate the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str):
        """Execute a command in the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return CLIResult(
                system="tool must be restarted",
                error=f"bash has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output = (
                        self._process.stdout._buffer.decode()
                    )  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # strip the sentinel and break
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        if output.endswith("\n"):
            output = output[:-1]

        error = (
            self._process.stderr._buffer.decode()
        )  # pyright: ignore[reportAttributeAccessIssue]
        if error.endswith("\n"):
            error = error[:-1]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return CLIResult(output=output, error=error)


class Bash(BaseTool):
    """Bash命令执行工具

    参数说明：
    - command: 支持三种模式：
        1. 正常bash命令
        2. 空字符串：获取运行中进程的输出
        3. 'ctrl+c'：终止当前进程

    异常处理：
    - ToolError: 包含会话状态异常和超时提示
    - 推荐通过restart参数重置异常会话

    典型使用场景：
    - 需要保持会话状态的CLI交互
    - 长期运行的进程监控
    - 需要精细控制超时的操作
    """

    name: str = "bash"
    description: str = _BASH_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.",
            },
        },
        "required": ["command"],
    }

    _session: Optional[_BashSession] = None

    async def execute(
        self, command: str | None = None, restart: bool = False, **kwargs
    ) -> CLIResult:
        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return CLIResult(system="tool has been restarted.")

        if self._session is None:
            self._session = _BashSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        raise ToolError("no command provided.")


if __name__ == "__main__":
    bash = Bash()
    rst = asyncio.run(bash.execute("ls -l"))
    print(rst)
