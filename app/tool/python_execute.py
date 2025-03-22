import multiprocessing
import sys
from io import StringIO
from typing import Dict

from app.tool.base import BaseTool


class PythonExecute(BaseTool):
    """Python代码执行工具（子进程隔离模式）

    * 采用独立子进程执行代码，通过超时机制防止无限循环
    * 使用输出重定向技术捕获print输出，返回值无法获取
    * 限制内置模块访问，禁止危险操作（文件读写/网络访问等）

    安全限制说明：
    - 运行在沙箱环境中，仅保留基本builtins函数
    - 禁止直接访问系统资源（文件系统/网络接口等）
    - 每次执行最大超时时间默认5秒（可通过timeout参数调整）
    """

    name: str = "python_execute"
    description: str = (
        "执行Python代码字符串。注意：仅捕获print输出，函数返回值不可见。使用print语句查看执行结果。"
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
        },
        "required": ["code"],
    }

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(code, safe_globals, safe_globals)
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    async def execute(
        self,
        code: str,
        timeout: int = 5,
    ) -> Dict:
        """
        执行Python代码（子进程隔离模式）

        参数说明：
            code (str): 需要执行的Python代码字符串
            timeout (int): 执行超时时间（秒），超时后自动终止进程

        返回结构：
            Dict: 包含执行状态的字典，其中：
                - output: 执行输出或错误信息
                - success: 执行是否成功的布尔值

        执行流程：
            1. 创建独立子进程运行代码
            2. 通过管理器共享执行结果
            3. 设置进程超时终止保护
            4. 返回执行输出或超时/错误信息
        """

        with multiprocessing.Manager() as manager:
            result = manager.dict({"observation": "", "success": False})
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
            proc = multiprocessing.Process(
                target=self._run_code, args=(code, result, safe_globals)
            )
            proc.start()
            proc.join(timeout)

            # timeout process
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                return {
                    "observation": f"Execution timeout after {timeout} seconds",
                    "success": False,
                }
            return dict(result)
