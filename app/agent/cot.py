from typing import Optional

from pydantic import Field

from app.agent.base import BaseAgent
from app.llm import LLM
from app.logger import logger
from app.prompt.cot_zh import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import AgentState, Message


class CoTAgent(BaseAgent):
    """Chain of Thought Agent - Focuses on demonstrating the thinking process of large language models without executing tools"""

    name: str = "cot"
    description: str = "An agent that uses Chain of Thought reasoning"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: Optional[str] = NEXT_STEP_PROMPT

    llm: LLM = Field(default_factory=LLM)

    max_steps: int = 1  # COT通常只需要一个步骤即可完成推理

    async def step(self) -> str:
        """执行一步思维链推理

        执行流程：
        1. 检查是否需要添加下一步提示词
        2. 调用语言模型进行推理
        3. 记录推理结果到记忆
        4. 设置完成状态

        返回：
            str: 推理结果文本
        """
        logger.info(f"🧠 {self.name}正在思考...")

        # 如果存在下一步提示词且不是首次对话，将提示词添加到用户消息中
        if self.next_step_prompt and len(self.messages) > 1:
            self.memory.add_message(Message.user_message(self.next_step_prompt))

        # 使用系统提示词和用户消息进行推理
        response = await self.llm.ask(
            messages=self.messages,
            system_msgs=(
                [Message.system_message(self.system_prompt)]
                if self.system_prompt
                else None
            ),
        )

        # 记录助手的回复到记忆中
        self.memory.add_message(Message.assistant_message(response))

        # 完成后设置状态为已完成
        self.state = AgentState.FINISHED

        return response
