from typing import List

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.swe import NEXT_STEP_TEMPLATE, SYSTEM_PROMPT
from app.tool import Bash, StrReplaceEditor, Terminate, ToolCollection


class SWEAgent(ToolCallAgent):
    """软件工程代理（专为代码工程任务优化）

    核心功能：
    - 集成Bash终端和代码编辑工具
    - 自动维护当前工作目录状态
    - 支持自然语言对话与代码执行的混合工作流
    """

    # 代理标识
    name: str = "swe"  # 代理名称（固定值）
    description: str = (
        "an autonomous AI programmer that interacts directly with the computer to solve tasks."  # 功能描述（保持英文）
    )

    # 提示词模板
    system_prompt: str = SYSTEM_PROMPT  # 系统指令模板（代码工程专用）
    next_step_prompt: str = NEXT_STEP_TEMPLATE  # 动态步骤提示模板（含目录变量）

    # 工具配置
    available_tools: ToolCollection = ToolCollection(
        Bash(), StrReplaceEditor(), Terminate()
    )  # 可用工具集（Bash/字符串替换编辑器/终止工具）
    special_tool_names: List[str] = Field(
        default_factory=lambda: [Terminate().name]
    )  # 特殊工具白名单

    # 执行参数
    max_steps: int = 30  # 覆盖父类默认步数限制

    # 运行时状态
    bash: Bash = Field(default_factory=Bash)  # Bash工具实例
    working_dir: str = "."  # 当前工作目录（动态更新）

    async def think(self) -> bool:
        """思考阶段实现（扩展自ToolCallAgent）

        执行流程：
        1. 通过pwd命令获取实时工作目录
        2. 动态更新next_step_prompt中的目录变量
        3. 调用父类think()完成基础思考逻辑
        """
        # 更新工作目录
        self.working_dir = await self.bash.execute("pwd")
        self.next_step_prompt = self.next_step_prompt.format(
            current_dir=self.working_dir
        )

        return await super().think()
