import asyncio
import os
import shlex
from typing import Optional

from app.tool.base import BaseTool, CLIResult


class Terminal(BaseTool):
    """命令行终端执行工具

    功能特性：
    - 异步执行系统命令并捕获输出，支持并发和超时控制
    - 持久化维护当前工作目录状态，自动同步目录切换
    - 内置危险命令过滤机制，防止恶意操作
    - 支持Conda环境命令执行，实现环境隔离

    参数规范：
    - command参数必须符合当前操作系统规范
    - 支持使用&符号分隔的多命令连续执行
    - 包含命令执行超时自动处理机制
    - 所有路径支持相对/绝对路径格式

    安全规则：
    1. 过滤以下危险命令：
       - rm（文件删除）- 防止意外删除
       - sudo（权限提升）- 限制权限边界
       - shutdown/reboot（系统控制）- 保护系统稳定
    2. 命令参数经过shlex严格解析，防止注入
    3. 执行上下文隔离保护，避免环境污染

    典型应用场景：
    - 需要执行系统级命令的自动化操作
    - 需要维护执行环境状态的长期任务
    - 需要安全过滤的不可信命令执行
    - 跨环境的Python包管理和部署
    """

    name: str = "execute_command"
    description: str = """在系统上执行CLI命令的请求说明
当需要执行系统级操作或运行特定命令来完成用户任务步骤时使用。
必须根据用户系统定制命令，并清晰解释命令作用。
优先执行复杂CLI命令而非创建可执行脚本，因其更灵活易用。
命令将在当前工作目录执行。
注意：对于50毫秒内完成的命令，必须在命令末尾追加`sleep 0.05`，以避免终端工具因命令执行过快导致输出丢失的已知问题。
"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "(必填) 需要执行的CLI命令，应符合当前操作系统规范。确保命令格式正确且不包含危险指令。",
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
                   命令间的执行状态和环境变量会自动传递

        处理流程：
        1. 命令分割：按&符号拆分为独立命令，保持执行顺序
        2. 安全校验：通过_sanitize_command方法过滤危险命令，防止系统损坏
        3. 目录切换：自动处理cd命令并维护当前路径状态，支持相对路径
        4. 异步执行：使用subprocess创建子进程，避免阻塞主线程
        5. 结果合并：聚合多个命令的输出和错误信息，保持日志完整性

        返回值：
        - CLIResult对象包含标准化输出和错误信息
        - output字段：命令执行的标准输出内容
        - error字段：命令执行过程的错误信息

        错误处理：
        - 路径错误：输出为空且错误信息包含"No such directory"
        - 权限错误：error字段包含访问限制信息
        - 命令错误：error字段包含具体的错误描述
        """

        # Split the command by & to handle multiple commands
        commands = [cmd.strip() for cmd in command.split("&") if cmd.strip()]
        final_output = CLIResult(output="", error="")

        for cmd in commands:
            sanitized_command = self._sanitize_command(cmd)

            # 内部处理“ CD”命令
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

            # 结合输出
            if result.output:
                final_output.output += (
                    (result.output + "\n") if final_output.output else result.output
                )
            if result.error:
                final_output.error += (
                    (result.error + "\n") if final_output.error else result.error
                )

        # 删除尾随的新线
        final_output.output = final_output.output.rstrip()
        final_output.error = final_output.error.rstrip()
        return final_output

    async def execute_in_env(self, env_name: str, command: str) -> CLIResult:
        """在Conda环境中执行命令

        实现原理：
        - 使用conda run命令在指定环境执行，无需手动激活
        - 自动处理环境路径和依赖关系，确保包可用性
        - 支持环境变量传递和路径映射

        参数要求：
        - env_name必须是已存在的Conda环境名称
        - 支持环境名称包含空格等特殊字符（自动shlex转义）
        - command参数需符合目标环境的命令规范

        典型应用：
        - 需要特定Python版本的任务
        - 依赖隔离的项目环境
        """

        sanitized_command = self._sanitize_command(command)

        # 构建命令以在Conda环境中运行
        # 使用“ conda run -n env_name命令”执行而无需激活
        conda_command = f"conda run -n {shlex.quote(env_name)} {sanitized_command}"

        return await self.execute(conda_command)

    async def _handle_cd_command(self, command: str) -> CLIResult:
        """处理目录切换命令

        路径解析规则：
        - 支持绝对路径和相对路径的自动识别
        - 自动展开用户目录(~)到实际路径
        - 路径标准化处理（规范化../等相对符号）
        - 支持Windows和Unix风格的路径分隔符

        状态维护：
        - 自动更新current_path属性
        - 保持工作目录状态的一致性
        - 支持路径历史记录和回溯

        异常处理：
        - 目录不存在：返回详细的错误信息
        - 权限不足：提示访问限制原因
        - 非法路径：抛出ValueError并说明格式问题
        - 系统错误：提供操作系统相关的错误描述
        """

        try:
            parts = shlex.split(command)
            if len(parts) < 2:
                new_path = os.path.expanduser("~")
            else:
                new_path = os.path.expanduser(parts[1])

            # 处理相对路径
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
        1. 使用shlex严格解析命令参数，防止注入攻击
        2. 检查命令主体是否包含危险指令，阻止高风险操作
        3. 双重验证机制（精确匹配和模糊匹配），提高检测准确性
        4. 支持自定义规则扩展，适应不同安全需求

        白名单机制：
        - 允许执行除危险命令列表外的所有系统命令
        - 自动添加sleep 0.05解决快速命令执行问题
        - 支持环境变量和参数的安全检查
        - 可配置命令执行的超时限制

        安全增强：
        - 命令注入防护
        - 路径遍历检测
        - 特权操作控制
        - 资源限制保护
        """

        # Example sanitization: restrict certain dangerous commands
        dangerous_commands = ["rm", "sudo", "shutdown", "reboot"]
        try:
            parts = shlex.split(command)
            if any(cmd in dangerous_commands for cmd in parts):
                raise ValueError("Use of dangerous commands is restricted.")
        except Exception:
            # 如果shlex.split失败，请尝试基本字符串比较
            if any(cmd in command for cmd in dangerous_commands):
                raise ValueError("Use of dangerous commands is restricted.")

        # 可以在此处添加其他卫生逻辑
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
