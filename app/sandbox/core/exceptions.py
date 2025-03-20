"""沙箱系统的异常类。

本模块定义了沙箱系统中使用的自定义异常，
用于以结构化的方式处理各种错误情况。
"""


class SandboxError(Exception):
    """沙箱相关错误的基础异常类。"""


class SandboxTimeoutError(SandboxError):
    """当沙箱操作超时时引发的异常。"""


class SandboxResourceError(SandboxError):
    """资源相关错误时引发的异常。"""
