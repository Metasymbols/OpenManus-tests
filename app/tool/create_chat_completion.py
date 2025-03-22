from typing import Any, List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, Field

from app.tool import BaseTool


class CreateChatCompletion(BaseTool):
    """结构化响应生成工具

    功能特性：
    - 根据指定类型生成符合JSON Schema规范的响应结构
    - 支持基础类型、Pydantic模型及容器类型的自动模式推导
    - 提供类型安全的结果转换机制

    类型映射机制：
    通过type_mapping将Python类型映射为JSON Schema类型，支持以下类型转换：
    - 基本类型：str -> string, int -> integer 等
    - 复杂类型：Pydantic模型自动生成完整Schema
    - 容器类型：List/Dict自动推导元素类型
    """

    name: str = "create_chat_completion"
    description: str = (
        "Creates a structured completion with specified output formatting."
    )

    # 类型映射表：Python类型到JSON Schema类型的映射
    type_mapping: dict = {
        str: "string",
        int: "integer",  # 整型数字
        float: "number",  # 浮点数字
        bool: "boolean",  # 布尔值
        dict: "object",  # 字典对象
        list: "array",  # 列表数组
    }
    response_type: Optional[Type] = None
    required: List[str] = Field(default_factory=lambda: ["response"])

    def __init__(self, response_type: Optional[Type] = str):
        """Initialize with a specific response type."""
        super().__init__()
        self.response_type = response_type
        self.parameters = self._build_parameters()

    def _build_parameters(self) -> dict:
        """Build parameters schema based on response type."""
        if self.response_type == str:
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The response text that should be delivered to the user.",
                    },
                },
                "required": self.required,
            }

        if isinstance(self.response_type, type) and issubclass(
            self.response_type, BaseModel
        ):
            schema = self.response_type.model_json_schema()
            return {
                "type": "object",
                "properties": schema["properties"],
                "required": schema.get("required", self.required),
            }

        return self._create_type_schema(self.response_type)

    def _create_type_schema(self, type_hint: Type) -> dict:
        """构建指定类型的JSON模式

        处理策略：
        1. 基础类型：直接映射为对应JSON类型
        2. 容器类型：递归处理元素类型
        3. 联合类型：生成anyOf组合模式

        参数：
            type_hint (Type): 需要解析的Python类型

        返回：
            dict: 符合JSON Schema规范的字典结构
        """
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # Handle primitive types
        if origin is None:
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": self.type_mapping.get(type_hint, "string"),
                        "description": f"Response of type {type_hint.__name__}",
                    }
                },
                "required": self.required,
            }

        # Handle List type
        if origin is list:
            item_type = args[0] if args else Any
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "array",
                        "items": self._get_type_info(item_type),
                    }
                },
                "required": self.required,
            }

        # Handle Dict type
        if origin is dict:
            value_type = args[1] if len(args) > 1 else Any
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "object",
                        "additionalProperties": self._get_type_info(value_type),
                    }
                },
                "required": self.required,
            }

        # Handle Union type
        if origin is Union:
            return self._create_union_schema(args)

        return self._build_parameters()

    def _get_type_info(self, type_hint: Type) -> dict:
        """获取单个类型的模式信息

        参数：
            type_hint (Type): 需要解析的类型对象

        返回：
            dict: 包含类型定义的模式字典
                  - 对Pydantic模型返回完整模型schema
                  - 基础类型返回类型映射结果
        """
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return type_hint.model_json_schema()

        return {
            "type": self.type_mapping.get(type_hint, "string"),
            "description": f"Value of type {getattr(type_hint, '__name__', 'any')}",
        }

    def _create_union_schema(self, types: tuple) -> dict:
        """创建联合类型的JSON模式

        参数：
            types (tuple): 多个类型组成的元组

        返回：
            dict: 包含anyOf结构的模式定义
        """
        return {
            "type": "object",
            "properties": {
                "response": {"anyOf": [self._get_type_info(t) for t in types]}
            },
            "required": self.required,
        }

    async def execute(self, required: list | None = None, **kwargs) -> Any:
        """执行聊天完成并处理类型转换

        参数：
            required (list | None): 必填字段列表，当包含多个字段时返回字典格式
            **kwargs: 响应数据

        返回：
            Any: 根据response_type转换后的响应结果

        类型转换策略：
        1. 字符串类型直接返回
        2. Pydantic模型实例化返回
        3. 容器类型保持原始格式
        4. 其他类型尝试强制类型转换
        """
        required = required or self.required

        # Handle case when required is a list
        if isinstance(required, list) and len(required) > 0:
            if len(required) == 1:
                required_field = required[0]
                result = kwargs.get(required_field, "")
            else:
                # Return multiple fields as a dictionary
                return {field: kwargs.get(field, "") for field in required}
        else:
            required_field = "response"
            result = kwargs.get(required_field, "")

        # Type conversion logic
        if self.response_type == str:
            return result

        if isinstance(self.response_type, type) and issubclass(
            self.response_type, BaseModel
        ):
            return self.response_type(**kwargs)

        if get_origin(self.response_type) in (list, dict):
            return result  # Assuming result is already in correct format

        try:
            return self.response_type(result)
        except (ValueError, TypeError):
            return result
