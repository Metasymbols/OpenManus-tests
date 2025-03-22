from app.tool.base import BaseTool


_TERMINATE_DESCRIPTION = """当满足请求条件或助手无法继续执行任务时终止交互
完成所有任务后调用此工具结束工作流程"""


class Terminate(BaseTool):
    """
    交互终止工具

    功能说明：
    - 在任务完成或无法继续执行时终止当前交互流程
    - 支持成功/失败两种状态标识执行结果

    触发条件：
    1. 成功状态标识(status=success)：
    - 所有任务目标已达成
    - 用户明确要求结束交互

    2. 失败状态标识(status=failure)：
    - 遇到无法解决的系统错误
    - 缺少必要执行上下文
    - 三次重试后仍无法推进任务
    """

    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "执行状态标识（成功/失败）",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    async def execute(self, **kwargs) -> str:
        """
        执行交互终止操作

        参数校验规则：
        - status必须为枚举值['success', 'failure']
        - 失败状态需确保错误日志已记录

        执行流程：
        1. 验证状态参数合法性
        """
        if not isinstance(kwargs, dict):
            raise TypeError("参数必须是字典类型")

        status = kwargs.get("status")
        if not status:
            raise ValueError("缺少必需的status参数")
        if status not in ["success", "failure"]:
            raise ValueError("status参数值无效，必须为'success'或'failure'之一")

        # 2. 清理临时资源
        # 3. 返回标准化终止响应
        return f"交互已完成，状态: {status}"
