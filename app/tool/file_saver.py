import os
from datetime import datetime

import aiofiles

from app.config import WORKSPACE_ROOT
from app.tool.base import BaseTool


class FileSaver(BaseTool):
    name: str = "file_saver"
    description: str = """将内容保存到指定路径的本地文件，自动添加时间戳防覆盖。

                        核心功能：
                        - 支持文本、代码及生成内容的持久化存储
                        - 自动处理文件路径转换（绝对路径转工作区相对路径）
                        - 异步写入确保高性能IO操作
                        - 完备的目录创建及错误处理机制

                        使用场景：
                        - 需要将生成内容保存到工作区时
                        - 需要追加内容到现有文件时
                        - 需要确保文件目录结构自动创建时
                        """
    parameters: dict = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "(必填) 需要保存的多行文本内容(需JSON转义特殊字符)，支持UNIX(LF)/Windows(CRLF)换行符自动转换",
                "format": "json-escaped-string",
            },
            "file_path": {
                "type": "string",
                "description": "(必填) 文件保存路径（含扩展名），支持绝对/相对路径。当文件存在时自动添加时间戳（格式：文件名_YYYYMMDD_HHMMSS.扩展名）",
            },
            "mode": {
                "type": "string",
                "description": "(可选) 文件打开模式，默认'w'覆盖写入，'a'追加写入。注意：追加模式需文件已存在",
                "enum": ["w", "a"],
                "default": "w",
            },
        },
        "required": ["content", "file_path"],
    }

    async def execute(self, content: str, file_path: str, mode: str = "w") -> str:
        try:
            # 将生成的文件放在工作区output目录
            output_dir = os.path.join(WORKSPACE_ROOT, "output")
            os.makedirs(output_dir, exist_ok=True)

            # 路径生成基础逻辑
            base_name = os.path.basename(file_path)
            if os.path.isabs(file_path):
                # 对于绝对路径，保留相对于WORKSPACE_ROOT的目录结构
                rel_path = os.path.relpath(file_path, WORKSPACE_ROOT)
                full_path = os.path.join(output_dir, rel_path)
            else:
                full_path = os.path.join(output_dir, file_path)

            # 防覆盖机制：添加时间戳
            if os.path.exists(full_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_part, ext_part = os.path.splitext(base_name)
                new_name = f"{name_part}_{timestamp}{ext_part}"
                full_path = os.path.join(os.path.dirname(full_path), new_name)

            # 确保目录存在
            directory = os.path.dirname(full_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # 内容格式预处理
            import json
            import re
            from typing import List, Tuple

            def validate_markdown(text: str) -> Tuple[bool, List[str]]:
                """验证Markdown格式并返回错误信息"""
                errors = []
                # 检查标题格式
                if re.search(r"^#{7,}\s", text, re.M):
                    errors.append("标题层级不应超过6级")
                # 检查代码块格式
                code_blocks = re.finditer(r"```(\w*)\n([\s\S]*?)```", text)
                for block in code_blocks:
                    lang = block.group(1)
                    if lang and not re.match(r"^[a-zA-Z0-9+#]+$", lang):
                        errors.append(f"无效的代码块语言标识: {lang}")
                return len(errors) == 0, errors

            def format_code_blocks(text: str) -> str:
                """格式化代码块内容"""

                def format_block(match) -> str:
                    lang = match.group(1)
                    code = match.group(2).strip()
                    # 规范化缩进
                    lines = code.split("\n")
                    if lines:
                        # 计算最小缩进
                        min_indent = float("inf")
                        for line in lines:
                            if line.strip():
                                indent = len(line) - len(line.lstrip())
                                min_indent = min(min_indent, indent)
                        if min_indent != float("inf"):
                            # 应用统一缩进
                            lines = [
                                line[min_indent:] if line.strip() else ""
                                for line in lines
                            ]
                        code = "\n".join(lines)
                    return f"```{lang}\n{code}\n```"

                return re.sub(r"```(\w*)\n([\s\S]*?)```", format_block, text)

            # 验证并格式化内容
            is_valid, errors = validate_markdown(content)
            if not is_valid:
                raise ValueError(f"Markdown格式错误:\n{chr(10).join(errors)}")

            # 格式化内容
            formatted_content = format_code_blocks(content)

            # JSON转义处理
            escaped_content = json.dumps(formatted_content)[1:-1]

            # 确保UTF-8编码
            try:
                normalized_content = escaped_content.encode("utf-8").decode("utf-8")
            except UnicodeError as e:
                raise ValueError(f"内容编码错误: {str(e)}")

            # 换行符标准化
            normalized_content = normalized_content.replace("\r\n", "\n").replace(
                "\r", "\n"
            )

            # 写入文件
            async with aiofiles.open(
                full_path, mode, encoding="utf-8", newline="\n"
            ) as file:
                await file.write(normalized_content)

            return f"Content successfully saved to {full_path}"
        except ValueError as ve:
            return f"格式验证错误: {str(ve)}"
        except Exception as e:
            return f"保存文件时发生错误: {str(e)}"
