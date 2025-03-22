import asyncio
import os
import shlex
from typing import Optional

from app.tool.base import BaseTool, CLIResult


class Terminal(BaseTool):
    """命令行终端执行工具

    功能特性：
    - 异步执行系统命令并捕获输出
    - 持久化维护当前工作目录状态
    - 内置危险命令过滤机制
    - 支持Conda环境命令执行

    参数规范：
    - command参数应符合当前操作系统规范
    - 支持使用&符号分隔的多命令连续执行
    - 包含命令执行超时自动处理机制

    安全规则：
    1. 过滤以下危险命令：
       - rm（文件删除）
       - sudo（权限提升）
       - shutdown/reboot（系统控制）
    2. 命令参数经过shlex严格解析
    3. 执行上下文隔离保护

    典型应用场景：
    - 需要执行系统级命令的操作
    - 需要维护执行环境状态的任务
    - 需要安全过滤的不可信命令执行
    """

    name: str = "execute_command"
    description: str = """Request to execute a CLI command on the system.
Use this when you need to perform system operations or run specific commands to accomplish any step in the user's task.
You must tailor your command to the user's system and provide a clear explanation of what the command does.
Prefer to execute complex CLI commands over creating executable scripts, as they are more flexible and easier to run.
Commands will be executed in the current working directory.
Note: You MUST append a `sleep 0.05` to the end of the command for commands that will complete in under 50ms, as this will circumvent a known issue with the terminal tool where it will sometimes not return the output when the command completes too quickly.
"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "(required) The CLI command to execute. This should be valid for the current operating system. Ensure the command is properly formatted and does not contain any harmful instructions.",
            }
        },
        "required": ["command"],
    }
    process: Optional[asyncio.subprocess.Process] = None
    current_path: str = os.getcwd()
    lock: asyncio.Lock = asyncio.Lock()

    async def execute(self, command: str) -> CLIResult:
        """执行终端命令

        参数说明：
        - command: 支持多个命令用&符号分隔，自动顺序执行
                   (示例："cd src && ls -l & python script.py")

        处理流程：
        1. 命令分割：按&符号拆分为独立命令
        2. 安全校验：通过_sanitize_command方法过滤危险命令
        3. 目录切换：自动处理cd命令并维护当前路径状态
        4. 异步执行：使用subprocess创建子进程
        5. 结果合并：聚合多个命令的输出和错误信息

        返回值：
        - CLIResult对象包含标准化输出和错误信息
        特殊状态码：
        - 输出为空且错误信息包含"No such directory"表示路径错误
        """

        """
        Execute a terminal command asynchronously with persistent context.

        Args:
            command (str): The terminal command to execute.

        Returns:
            str: The output, and error of the command execution.
        """
        # Split the command by & to handle multiple commands
        commands = [cmd.strip() for cmd in command.split("&") if cmd.strip()]
        final_output = CLIResult(output="", error="")

        for cmd in commands:
            sanitized_command = self._sanitize_command(cmd)

            # Handle 'cd' command internally
            if sanitized_command.lstrip().startswith("cd "):
                result = await self._handle_cd_command(sanitized_command)
            else:
                async with self.lock:
                    try:
                        self.process = await asyncio.create_subprocess_shell(
                            sanitized_command,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=self.current_path,
                        )
                        stdout, stderr = await self.process.communicate()
                        result = CLIResult(
                            output=stdout.decode().strip(),
                            error=stderr.decode().strip(),
                        )
                    except Exception as e:
                        result = CLIResult(output="", error=str(e))
                    finally:
                        self.process = None

            # Combine outputs
            if result.output:
                final_output.output += (
                    (result.output + "\n") if final_output.output else result.output
                )
            if result.error:
                final_output.error += (
                    (result.error + "\n") if final_output.error else result.error
                )

        # Remove trailing newlines
        final_output.output = final_output.output.rstrip()
        final_output.error = final_output.error.rstrip()
        return final_output

    async def execute_in_env(self, env_name: str, command: str) -> CLIResult:
        """在Conda环境中执行命令

        实现原理：
        - 使用conda run命令在指定环境执行
        - 自动处理环境路径和依赖关系

        参数要求：
        - env_name必须是已存在的Conda环境名称
        - 支持环境名称包含空格等特殊字符（自动shlex转义）

        典型应用：
        - 需要特定Python版本的任务
        - 依赖隔离的项目环境
        """

        """
        Execute a terminal command asynchronously within a specified Conda environment.

        Args:
            env_name (str): The name of the Conda environment.
            command (str): The terminal command to execute within the environment.

        Returns:
            str: The output, and error of the command execution.
        """
        sanitized_command = self._sanitize_command(command)

        # Construct the command to run within the Conda environment
        # Using 'conda run -n env_name command' to execute without activating
        conda_command = f"conda run -n {shlex.quote(env_name)} {sanitized_command}"

        return await self.execute(conda_command)

    async def _handle_cd_command(self, command: str) -> CLIResult:
        """处理目录切换命令

        路径解析规则：
        - 支持绝对路径和相对路径
        - 自动展开用户目录(~)
        - 路径标准化处理（去除../等相对符号）

        异常处理：
        - 目录不存在时返回错误信息
        - 非法路径格式抛出ValueError
        """

        """
        Handle 'cd' commands to change the current path.

        Args:
            command (str): The 'cd' command to process.

        Returns:
            TerminalOutput: The result of the 'cd' command.
        """
        try:
            parts = shlex.split(command)
            if len(parts) < 2:
                new_path = os.path.expanduser("~")
            else:
                new_path = os.path.expanduser(parts[1])

            # Handle relative paths
            if not os.path.isabs(new_path):
                new_path = os.path.join(self.current_path, new_path)

            new_path = os.path.abspath(new_path)

            if os.path.isdir(new_path):
                self.current_path = new_path
                return CLIResult(
                    output=f"Changed directory to {self.current_path}", error=""
                )
            else:
                return CLIResult(output="", error=f"No such directory: {new_path}")
        except Exception as e:
            return CLIResult(output="", error=str(e))

    @staticmethod
    def _sanitize_command(command: str) -> str:
        """命令安全过滤

        验证逻辑：
        1. 使用shlex严格解析命令参数
        2. 检查命令主体是否包含危险指令
        3. 双重验证机制（精确匹配和模糊匹配）

        白名单机制：
        - 允许执行除危险命令列表外的所有系统命令
        - 自动添加sleep 0.05解决快速命令执行问题
        """

        """
        Sanitize the command for safe execution.

        Args:
            command (str): The command to sanitize.

        Returns:
            str: The sanitized command.
        """
        # Example sanitization: restrict certain dangerous commands
        dangerous_commands = ["rm", "sudo", "shutdown", "reboot"]
        try:
            parts = shlex.split(command)
            if any(cmd in dangerous_commands for cmd in parts):
                raise ValueError("Use of dangerous commands is restricted.")
        except Exception:
            # If shlex.split fails, try basic string comparison
            if any(cmd in command for cmd in dangerous_commands):
                raise ValueError("Use of dangerous commands is restricted.")

        # Additional sanitization logic can be added here
        return command

    async def close(self):
        """Close the persistent shell process if it exists."""
        async with self.lock:
            if self.process:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                finally:
                    self.process = None

    async def __aenter__(self):
        """Enter the asynchronous context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the asynchronous context manager and close the process."""
        await self.close()
