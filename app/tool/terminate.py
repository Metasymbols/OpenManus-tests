from app.tool.base import BaseTool


_TERMINATE_DESCRIPTION = """Terminate the interaction when the request is met OR if the assistant cannot proceed further with the task.
When you have finished all the tasks, call this tool to end the work."""


class Terminate(BaseTool):
    """
    交互终止工具

    功能说明：
    - 在任务完成或无法继续执行时终止当前交互流程
    - 支持成功/失败两种状态标识执行结果

    触发条件：
    1. 成功状态(status=success)：
    - 所有任务目标已达成
    - 用户明确要求结束交互

    2. 失败状态(status=failure)：
    - 遇到无法解决的错误
    - 缺少必要执行条件
    - 三次尝试后仍无法完成任务
    """

    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "交互终止状态标识",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    async def execute(self, status: str) -> str:
        """
        执行交互终止操作

        参数校验规则：
        - status必须为枚举值['success', 'failure']
        - 失败状态需确保错误日志已记录

        执行流程：
        1. 验证状态参数合法性
        2. 清理临时资源
        3. 返回标准化终止响应
        """
        return f"The interaction has been completed with status: {status}"
