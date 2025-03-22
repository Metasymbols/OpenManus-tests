# tool/planning.py
from typing import Dict, List, Literal, Optional

from app.exceptions import ToolError
from app.tool.base import BaseTool, ToolResult


_PLANNING_TOOL_DESCRIPTION = """
计划管理工具，用于创建和管理复杂任务的执行方案

核心能力：
- 多步骤任务的全生命周期管理（创建/更新/删除）
- 步骤状态跟踪（未开始/进行中/完成/阻塞）
- 活动计划切换与状态持久化存储
- 可视化进度展示与完成度统计

使用场景：
- 将复杂工作流拆解为可执行的步骤序列
- 长期项目的阶段进度跟踪与管理
- 团队协作时的任务状态同步与监控
- 需要可视化展示执行进度的自动化任务
"""


class PlanningTool(BaseTool):
    """
    计划管理工具类，用于创建和管理复杂任务的执行计划

    核心功能：
    - 计划的创建、更新、删除全生命周期管理
    - 多步骤任务的状态跟踪（未开始/进行中/完成/阻塞）
    - 活动计划的切换与状态持久化
    - 计划执行的进度可视化展示

    典型使用场景：
    - 复杂任务拆解为可执行的步骤序列
    - 多阶段项目的进度跟踪
    - 协作任务的状态同步与监控
    """

    name: str = "planning"
    description: str = _PLANNING_TOOL_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "执行的操作命令，可选值：\n- create: 创建新计划\n- update: 更新现有计划\n- list: 列出所有计划\n- get: 获取计划详情\n- set_active: 设置活动计划\n- mark_step: 标记步骤状态\n- delete: 删除计划",
                "enum": [
                    "create",
                    "update",
                    "list",
                    "get",
                    "set_active",
                    "mark_step",
                    "delete",
                ],
                "type": "string",
            },
            "plan_id": {
                "description": "计划唯一标识符\n- create/update/set_active/delete命令必填\n- get/mark_step命令可选（未提供时使用当前活动计划）\n- 格式要求：字母开头，支持字母、数字和下划线",
                "type": "string",
                "pattern": "^[A-Za-z][A-Za-z0-9_]*$",
                "examples": ["project_x", "backend_refactor"],
            },
            "title": {
                "description": "计划标题（用于可视化展示）\n- create命令必填\n- update命令可选\n- 长度限制：2-64个字符",
                "type": "string",
                "minLength": 2,
                "maxLength": 64,
                "examples": ["产品发布计划", "后端服务重构方案"],
            },
            "steps": {
                "description": "计划步骤列表\n- create命令必须包含至少1个步骤\n- update命令可选（更新时保留已有步骤状态）\n- 每个步骤应为明确的动作描述",
                "type": "array",
                "items": {
                    "type": "string",
                    "minLength": 3,
                    "examples": ["完成需求评审", "部署测试环境"],
                },
                "minItems": 1,
            },
            "step_index": {
                "description": "要更新的步骤索引（从0开始）\n- mark_step命令必填\n- 必须小于步骤总数",
                "type": "integer",
                "minimum": 0,
                "examples": [0, 2],
            },
            "step_status": {
                "description": "步骤状态设置规则：\n- 已完成(completed)不可回退为未开始(not_started)\n- 阻塞状态(blocked)必须配合step_notes说明原因\n- 进行中(in_progress)会自动继承上一步完成状态",
                "enum": ["not_started", "in_progress", "completed", "blocked"],
                "type": "string",
                "examples": ["in_progress", "blocked"],
            },
            "step_notes": {
                "description": "步骤状态注释说明\n- 阻塞状态必须提供说明\n- 进行中状态建议提供进度说明\n- 最大长度：200字符",
                "type": "string",
                "maxLength": 200,
                "examples": ["等待测试环境就绪", "完成80%代码重构"],
            },
        },
        "required": ["command"],
        "additionalProperties": False,
        "dependencies": {
            "command": {
                "create": ["plan_id", "title", "steps"],
                "update": ["plan_id"],
                "mark_step": ["step_index"],
                "set_active": ["plan_id"],
                "delete": ["plan_id"],
            }
        },
    }

    plans: dict = {}  # Dictionary to store plans by plan_id
    _current_plan_id: Optional[str] = None  # Track the current active plan

    async def execute(
        self,
        *,
        command: Literal[
            "create", "update", "list", "get", "set_active", "mark_step", "delete"
        ],
        plan_id: Optional[str] = None,
        title: Optional[str] = None,
        steps: Optional[List[str]] = None,
        step_index: Optional[int] = None,
        step_status: Optional[
            Literal["not_started", "in_progress", "completed", "blocked"]
        ] = None,
        step_notes: Optional[str] = None,
        **kwargs,
    ):
        """
        Execute the planning tool with the given command and parameters.

        Parameters:
        - command: The operation to perform
        - plan_id: Unique identifier for the plan
        - title: Title for the plan (used with create command)
        - steps: List of steps for the plan (used with create command)
        - step_index: Index of the step to update (used with mark_step command)
        - step_status: Status to set for a step (used with mark_step command)
        - step_notes: Additional notes for a step (used with mark_step command)
        """

        if command == "create":
            return self._create_plan(plan_id, title, steps)
        elif command == "update":
            return self._update_plan(plan_id, title, steps)
        elif command == "list":
            return self._list_plans()
        elif command == "get":
            return self._get_plan(plan_id)
        elif command == "set_active":
            return self._set_active_plan(plan_id)
        elif command == "mark_step":
            return self._mark_step(plan_id, step_index, step_status, step_notes)
        elif command == "delete":
            return self._delete_plan(plan_id)
        else:
            raise ToolError(
                f"Unrecognized command: {command}. Allowed commands are: create, update, list, get, set_active, mark_step, delete"
            )

    def _create_plan(
        self, plan_id: Optional[str], title: Optional[str], steps: Optional[List[str]]
    ) -> ToolResult:
        """
        创建新计划

        参数：
        - plan_id (str): 计划唯一标识符，必填
        - title (str): 计划标题，用于可视化展示，必填
        - steps (List[str]): 计划步骤列表，至少包含一个步骤

        返回：
        ToolResult: 包含计划ID和格式化计划详情的工具结果

        示例：
        >>> create_plan("project_x", "产品发布计划", ["需求评审", "开发", "测试", "上线"])
        """
        if not plan_id:
            raise ToolError("Parameter `plan_id` is required for command: create")

        if plan_id in self.plans:
            raise ToolError(
                f"A plan with ID '{plan_id}' already exists. Use 'update' to modify existing plans."
            )

        if not title:
            raise ToolError("Parameter `title` is required for command: create")

        if (
            not steps
            or not isinstance(steps, list)
            or not all(isinstance(step, str) for step in steps)
        ):
            raise ToolError(
                "Parameter `steps` must be a non-empty list of strings for command: create"
            )

        # Create a new plan with initialized step statuses
        plan = {
            "plan_id": plan_id,
            "title": title,
            "steps": steps,
            "step_statuses": ["not_started"] * len(steps),
            "step_notes": [""] * len(steps),
        }

        self.plans[plan_id] = plan
        self._current_plan_id = plan_id  # Set as active plan

        return ToolResult(
            output=f"Plan created successfully with ID: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _update_plan(
        self, plan_id: Optional[str], title: Optional[str], steps: Optional[List[str]]
    ) -> ToolResult:
        """
        更新现有计划

        参数：
        - plan_id (str): 要更新的计划ID，必填
        - title (str): 新标题（可选，保留原值若不提供）
        - steps (List[str]): 新步骤列表（可选，保留原值若不提供）

        更新规则：
        - 步骤列表更新时保留原有步骤状态
        - 新增步骤状态初始化为'未开始'
        - 被删除步骤的状态信息将被永久移除
        """
        if not plan_id:
            raise ToolError("Parameter `plan_id` is required for command: update")

        if plan_id not in self.plans:
            raise ToolError(f"No plan found with ID: {plan_id}")

        plan = self.plans[plan_id]

        if title:
            plan["title"] = title

        if steps:
            if not isinstance(steps, list) or not all(
                isinstance(step, str) for step in steps
            ):
                raise ToolError(
                    "Parameter `steps` must be a list of strings for command: update"
                )

            # Preserve existing step statuses for unchanged steps
            old_steps = plan["steps"]
            old_statuses = plan["step_statuses"]
            old_notes = plan["step_notes"]

            # Create new step statuses and notes
            new_statuses = []
            new_notes = []

            for i, step in enumerate(steps):
                # If the step exists at the same position in old steps, preserve status and notes
                if i < len(old_steps) and step == old_steps[i]:
                    new_statuses.append(old_statuses[i])
                    new_notes.append(old_notes[i])
                else:
                    new_statuses.append("not_started")
                    new_notes.append("")

            plan["steps"] = steps
            plan["step_statuses"] = new_statuses
            plan["step_notes"] = new_notes

        return ToolResult(
            output=f"Plan updated successfully: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _list_plans(self) -> ToolResult:
        """
        列出所有存储的计划

        返回内容：
        - 计划ID列表及完成进度
        - 当前活动计划标记
        - 各计划的步骤完成统计

        示例输出：
        • project_x (active): 产品发布计划 - 2/4 steps completed
        • backend_refactor: 后端重构计划 - 0/5 steps completed
        """
        if not self.plans:
            return ToolResult(
                output="No plans available. Create a plan with the 'create' command."
            )

        output = "Available plans:\n"
        for plan_id, plan in self.plans.items():
            current_marker = " (active)" if plan_id == self._current_plan_id else ""
            completed = sum(
                1 for status in plan["step_statuses"] if status == "completed"
            )
            total = len(plan["steps"])
            progress = f"{completed}/{total} steps completed"
            output += f"• {plan_id}{current_marker}: {plan['title']} - {progress}\n"

        return ToolResult(output=output)

    def _get_plan(self, plan_id: Optional[str]) -> ToolResult:
        """
        获取计划完整详情

        参数：
        - plan_id (str): 可选，未提供时返回当前活动计划

        返回包含：
        - 计划标题与ID
        - 带状态标记的步骤列表
        - 步骤注释说明
        - 可视化进度条
        - 各状态步骤统计
        """
        if not plan_id:
            # If no plan_id is provided, use the current active plan
            if not self._current_plan_id:
                raise ToolError(
                    "No active plan. Please specify a plan_id or set an active plan."
                )
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            raise ToolError(f"No plan found with ID: {plan_id}")

        plan = self.plans[plan_id]
        return ToolResult(output=self._format_plan(plan))

    def _set_active_plan(self, plan_id: Optional[str]) -> ToolResult:
        """
        设置当前活动计划

        参数：
        - plan_id (str): 要激活的计划ID

        影响：
        - 后续无plan_id参数的操作默认使用此活动计划
        - 可视化展示优先显示活动计划
        - 同一时间只能有一个活动计划
        """
        if not plan_id:
            raise ToolError("Parameter `plan_id` is required for command: set_active")

        if plan_id not in self.plans:
            raise ToolError(f"No plan found with ID: {plan_id}")

        self._current_plan_id = plan_id
        return ToolResult(
            output=f"Plan '{plan_id}' is now the active plan.\n\n{self._format_plan(self.plans[plan_id])}"
        )

    def _mark_step(
        self,
        plan_id: Optional[str],
        step_index: Optional[int],
        step_status: Optional[str],
        step_notes: Optional[str],
    ) -> ToolResult:
        """
        标记步骤状态

        参数：
        - step_index (int): 要更新的步骤索引(0起始)
        - step_status (str): 新状态值，可选
        - step_notes (str): 状态注释说明，可选

        状态变更规则：
        1. 已完成步骤不可回退至未开始
        2. 阻塞状态需提供注释说明
        3. 进行中状态自动继承上一步完成状态
        """
        if not plan_id:
            # If no plan_id is provided, use the current active plan
            if not self._current_plan_id:
                raise ToolError(
                    "No active plan. Please specify a plan_id or set an active plan."
                )
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            raise ToolError(f"No plan found with ID: {plan_id}")

        if step_index is None:
            raise ToolError("Parameter `step_index` is required for command: mark_step")

        plan = self.plans[plan_id]

        if step_index < 0 or step_index >= len(plan["steps"]):
            raise ToolError(
                f"Invalid step_index: {step_index}. Valid indices range from 0 to {len(plan['steps'])-1}."
            )

        if step_status and step_status not in [
            "not_started",
            "in_progress",
            "completed",
            "blocked",
        ]:
            raise ToolError(
                f"Invalid step_status: {step_status}. Valid statuses are: not_started, in_progress, completed, blocked"
            )

        if step_status:
            plan["step_statuses"][step_index] = step_status

        if step_notes:
            plan["step_notes"][step_index] = step_notes

        return ToolResult(
            output=f"Step {step_index} updated in plan '{plan_id}'.\n\n{self._format_plan(plan)}"
        )

    def _delete_plan(self, plan_id: Optional[str]) -> ToolResult:
        """
        永久删除指定计划

        注意：
        - 删除操作不可逆
        - 如果删除的是当前活动计划，会自动清除活动状态
        - 关联的步骤跟踪数据将全部移除

        参数：
        - plan_id (str): 要删除的计划ID
        """
        if not plan_id:
            raise ToolError("Parameter `plan_id` is required for command: delete")

        if plan_id not in self.plans:
            raise ToolError(f"No plan found with ID: {plan_id}")

        del self.plans[plan_id]

        # If the deleted plan was the active plan, clear the active plan
        if self._current_plan_id == plan_id:
            self._current_plan_id = None

        return ToolResult(output=f"Plan '{plan_id}' has been deleted.")

    def _format_plan(self, plan: Dict) -> str:
        """
        格式化计划详情展示

        输出规范：
        1. 计划标题与ID显式标注
        2. 进度统计包含完成率/各状态计数
        3. 步骤列表使用符号系统：
           [ ] 未开始 | [→] 进行中 | [✓] 已完成 | [!] 阻塞
        4. 带注释的步骤显示额外缩进说明

        返回：
        str: 结构化的可视化计划字符串
        """
        output = f"Plan: {plan['title']} (ID: {plan['plan_id']})\n"
        output += "=" * len(output) + "\n\n"

        # Calculate progress statistics
        total_steps = len(plan["steps"])
        completed = sum(1 for status in plan["step_statuses"] if status == "completed")
        in_progress = sum(
            1 for status in plan["step_statuses"] if status == "in_progress"
        )
        blocked = sum(1 for status in plan["step_statuses"] if status == "blocked")
        not_started = sum(
            1 for status in plan["step_statuses"] if status == "not_started"
        )

        output += f"Progress: {completed}/{total_steps} steps completed "
        if total_steps > 0:
            percentage = (completed / total_steps) * 100
            output += f"({percentage:.1f}%)\n"
        else:
            output += "(0%)\n"

        output += f"Status: {completed} completed, {in_progress} in progress, {blocked} blocked, {not_started} not started\n\n"
        output += "Steps:\n"

        # Add each step with its status and notes
        for i, (step, status, notes) in enumerate(
            zip(plan["steps"], plan["step_statuses"], plan["step_notes"])
        ):
            status_symbol = {
                "not_started": "[ ]",
                "in_progress": "[→]",
                "completed": "[✓]",
                "blocked": "[!]",
            }.get(status, "[ ]")

            output += f"{i}. {status_symbol} {step}\n"
            if notes:
                output += f"   Notes: {notes}\n"

        return output
