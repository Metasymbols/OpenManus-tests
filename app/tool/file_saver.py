import os

import aiofiles

from app.config import WORKSPACE_ROOT
from app.tool.base import BaseTool


class FileSaver(BaseTool):
    name: str = "file_saver"
    description: str = """将内容保存到指定路径的本地文件。

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
                "description": "(必填) 需要保存到文件的内容文本，支持多行格式",
            },
            "file_path": {
                "type": "string",
                "description": "(必填) 文件保存路径，需包含文件名和扩展名。支持绝对路径或相对output目录的路径（相对路径基于工作区output子目录）",
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
        """
        将内容异步保存到指定路径的文件。

        实现流程：
        1. 路径规范化处理：
           - 绝对路径提取文件名，结合WORKSPACE_ROOT生成最终路径
           - 相对路径直接拼接工作区根目录
        2. 目录自动创建：检查并递归创建缺失的目录结构
        3. 异步写入：使用aiofiles库实现非阻塞文件操作
        4. 异常处理：捕获所有IO相关异常并返回友好提示

        参数说明：
            content (str): 需要保存的文本内容，支持多行文本格式
            file_path (str): 文件存储路径（支持绝对/相对路径格式）
            mode (str, optional): 文件写入模式，默认覆盖('w')，可选追加('a')

        返回：
            str: 操作结果信息，包含成功路径或错误详情

        错误处理：
            - 捕获OSError处理目录创建失败
            - 捕获IOError处理文件写入异常
            - 返回包含错误描述的友好提示信息
        """
        try:
            # 将生成的文件放在工作区output目录
            output_dir = os.path.join(WORKSPACE_ROOT, "output")
            os.makedirs(output_dir, exist_ok=True)

            # 统一路径处理逻辑
            if os.path.isabs(file_path):
                # 转换绝对路径为工作区相对路径
                try:
                    rel_path = os.path.relpath(file_path, WORKSPACE_ROOT)
                    if rel_path.startswith(".."):
                        raise ValueError("Path outside workspace")
                except ValueError as e:
                    self.logger.warning(f"Invalid path conversion: {str(e)}")
                    rel_path = os.path.basename(file_path)
                full_path = os.path.join(output_dir, rel_path)
            else:
                full_path = os.path.join(output_dir, file_path)

            # 递归创建目标目录并记录调试信息
            target_dir = os.path.dirname(full_path)
            if not os.path.exists(target_dir):
                self.logger.debug(f"Creating directory: {target_dir}")
                os.makedirs(target_dir, exist_ok=True)

            # 异步上下文管理器增强资源管理
            async with aiofiles.open(full_path, mode, encoding="utf-8") as file:
                await file.write(content)
                await file.flush()

            return f"内容已成功保存至 {full_path}"

        except Exception as e:
            self.logger.error(f"File operation failed: {str(e)}", exc_info=True)
            return f"文件保存失败: {str(e)}"
