"""Utility to run shell commands asynchronously with a timeout."""

import asyncio


TRUNCATED_MESSAGE: str = (
    "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
)
MAX_RESPONSE_LEN: int = 16000


def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN):
    """Truncate content and append a notice if content exceeds the specified length."""
    return (
        content
        if not truncate_after or len(content) <= truncate_after
        else content[:truncate_after] + TRUNCATED_MESSAGE
    )


async def run(
    cmd: str,
    timeout: float | None = 120.0,  # seconds
    truncate_after: int | None = MAX_RESPONSE_LEN,
):
    """异步执行Shell命令的核⼼函数

    功能特性：
    - 支持带超时机制的异步命令执行
    - 自动处理输出截断以避免内存溢出
    - 完善的进程生命周期管理

    参数说明：
    - cmd:       需要执行的Shell命令字符串
    - timeout:   超时时间（秒），默认120秒，设为None禁用超时
    - truncate_after: 输出截断长度（字符数），默认16000字符

    返回值：
    - tuple: (退出码, 标准输出, 标准错误)

    异常处理：
    - TimeoutError: 命令执行超时时抛出，包含超时详细信息
    - 自动清理：超时或异常时自动终止关联进程

    执行流程：
    1. 创建异步子进程执行命令
    2. 启动带超时控制的通信等待
    3. 对输出进行安全截断处理
    4. 返回标准化执行结果

    典型应用场景：
    - 需要严格限制执行时间的危险操作
    - 处理可能产生大量输出的长时间任务
    - 作为其他工具类的基础执行引擎
    """
    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return (
            process.returncode or 0,
            maybe_truncate(stdout.decode(), truncate_after=truncate_after),
            maybe_truncate(stderr.decode(), truncate_after=truncate_after),
        )
    except asyncio.TimeoutError as exc:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        raise TimeoutError(
            f"Command '{cmd}' timed out after {timeout} seconds"
        ) from exc
