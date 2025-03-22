"""File and directory manipulation tool with sandbox support."""

from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Any, DefaultDict, List, Literal, Optional, get_args

from app.config import config
from app.exceptions import ToolError
from app.tool import BaseTool
from app.tool.base import CLIResult, ToolResult
from app.tool.file_operators import FileOperator, LocalFileOperator, SandboxFileOperator


Command = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "undo_edit",
]

# Constants
SNIPPET_LINES: int = 4
MAX_RESPONSE_LEN: int = 16000

TRUNCATED_MESSAGE: str = (
    "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"  # 内容截断提示常量
)
# 触发条件：响应内容超过16000字符时自动激活
# 处理建议：
# 1. 使用`search_by_keyword`工具定位关键词所在行号
# 2. 通过`view`命令的view_range参数指定精确行号范围
# 3. 路径必须符合验证规则（绝对路径、有效存在）才能正确重试

_STR_REPLACE_EDITOR_DESCRIPTION = """文件系统操作工具，支持查看、创建和编辑文件
* 状态在多个命令调用和用户对话间保持持久化
* 路径验证规则：
  - `path`必须是绝对路径
  - `create`命令路径不能已存在
  - 目录路径只能使用`view`命令
* 内容截断策略：
  - 超过16000字符的响应会被截断并标记`<response clipped>`
  - 文件查看默认显示完整内容，可通过`view_range`指定行号范围
* 命令说明：
  - `view`: 查看文件内容(带行号)或目录结构(最多2层)
  - `create`: 创建新文件，需提供完整文件内容
  - `str_replace`: 精确替换文件内容，要求旧字符串全局唯一
  - `insert`: 在指定行号后插入新内容
  - `undo_edit`: 撤销最近一次文件修改

字符串替换操作规范：
1. `old_str`必须与源文件内容完全匹配(包括空白符)
2. 出现多次匹配时会拒绝执行替换
3. `new_str`需包含完整的替换内容，支持多行文本
3. 替换后会保留操作历史供撤销使用
"""


def maybe_truncate(
    content: str, truncate_after: Optional[int] = MAX_RESPONSE_LEN
) -> str:
    """Truncate content and append a notice if content exceeds the specified length."""
    if not truncate_after or len(content) <= truncate_after:
        return content
    return content[:truncate_after] + TRUNCATED_MESSAGE


class StrReplaceEditor(BaseTool):
    """A tool for viewing, creating, and editing files with sandbox support."""

    name: str = "str_replace_editor"
    description: str = _STR_REPLACE_EDITOR_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                "type": "string",
            },
            "path": {
                "description": "Absolute path to file or directory.",
                "type": "string",
            },
            "file_text": {
                "description": "Required parameter of `create` command, with the content of the file to be created.",
                "type": "string",
            },
            "old_str": {
                "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                "type": "string",
            },
            "new_str": {
                "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
                "type": "string",
            },
            "insert_line": {
                "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                "type": "integer",
            },
            "view_range": {
                "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                "items": {"type": "integer"},
                "type": "array",
            },
        },
        "required": ["command", "path"],
    }
    _file_history: DefaultDict[PathLike, List[str]] = defaultdict(list)
    _local_operator: LocalFileOperator = LocalFileOperator()
    _sandbox_operator: SandboxFileOperator = SandboxFileOperator()

    # def _get_operator(self, use_sandbox: bool) -> FileOperator:
    def _get_operator(self) -> FileOperator:
        """Get the appropriate file operator based on execution mode."""
        return (
            self._sandbox_operator
            if config.sandbox.use_sandbox
            else self._local_operator
        )

    async def execute(
        self,
        *,
        command: Command,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute a file operation command."""
        # Get the appropriate file operator
        operator = self._get_operator()

        # Validate path and command combination
        await self.validate_path(command, Path(path), operator)

        # Execute the appropriate command
        if command == "view":
            result = await self.view(path, view_range, operator)
        elif command == "create":
            if file_text is None:
                raise ToolError("Parameter `file_text` is required for command: create")
            await operator.write_file(path, file_text)
            self._file_history[path].append(file_text)
            result = ToolResult(output=f"File created successfully at: {path}")
        elif command == "str_replace":
            if old_str is None:
                raise ToolError(
                    "Parameter `old_str` is required for command: str_replace"
                )
            result = await self.str_replace(path, old_str, new_str, operator)
        elif command == "insert":
            if insert_line is None:
                raise ToolError(
                    "Parameter `insert_line` is required for command: insert"
                )
            if new_str is None:
                raise ToolError("Parameter `new_str` is required for command: insert")
            result = await self.insert(path, insert_line, new_str, operator)
        elif command == "undo_edit":
            result = await self.undo_edit(path, operator)
        else:
            # This should be caught by type checking, but we include it for safety
            raise ToolError(
                f"无法识别的命令 {command}。{self.name} 工具允许的命令有: {', '.join(get_args(Command))}"
            )

        return str(result)

    async def validate_path(self, command: str, path: Path, operator: FileOperator):
        """路径验证逻辑

        执行命令前验证路径合法性，包括：
        1. 绝对路径检查
        2. 路径存在性检查
        3. 文件/目录类型与命令的兼容性

        Args:
            command: 当前执行的命令名称
            path: 待验证的Path对象

        Raises:
            ToolError: 当出现以下情况时抛出异常：
                - 路径非绝对路径
                - 路径不存在且命令非create
                - 目录路径使用非view命令
                - 已存在路径执行create命令
        """
        # Check if its an absolute path
        if not path.is_absolute():
            suggested_path = Path("") / path
            raise ToolError(
                f"路径 {path} 不是绝对路径，路径应以 '/' 开头。您是否想要使用 {suggested_path}？"
            )
        # Check if path exists
        if not path.exists() and command != "create":
            raise ToolError(f"路径 {path} 不存在。请提供有效的路径。")
        if path.exists() and command == "create":
            raise ToolError(
                f"文件已存在于: {path}。不能使用 'create' 命令覆盖现有文件。"
            )
        # Check if the path points to a directory
        if path.is_dir():
            if command != "view":
                raise ToolError(f"路径 {path} 是一个目录，只能对目录使用 'view' 命令")

    async def view(self, path: Path, view_range: list[int] | None = None):
        """Implement the view command"""
        if path.is_dir():
            if view_range:
                raise ToolError("当路径指向目录时，不允许使用 'view_range' 参数。")

            _, stdout, stderr = await run(
                rf"find {path} -maxdepth 2 -not -path '*/\.*'"
            )
            if not stderr:
                stdout = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
            return CLIResult(output=stdout, error=stderr)

    async def _view_file(
        self,
        path: PathLike,
        operator: FileOperator,
        view_range: Optional[List[int]] = None,
    ) -> CLIResult:
        """Display file content, optionally within a specified line range."""
        # Read file content
        file_content = await operator.read_file(path)
        init_line = 1

        # Apply view range if specified
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise ToolError("无效的 'view_range'。它应该是一个包含两个整数的列表。")
            file_lines = file_content.split("\n")
            n_lines_file = len(file_lines)
            init_line, final_line = view_range

            # Validate view range
            if init_line < 1 or init_line > n_lines_file:
                raise ToolError(
                    f"无效的 'view_range': {view_range}。第一个元素 '{init_line}' 应该在文件的行数范围内: {[1, n_lines_file]}"
                )
            if final_line > n_lines_file:
                raise ToolError(
                    f"无效的 'view_range': {view_range}。第二个元素 '{final_line}' 应该小于文件的总行数: '{n_lines_file}'"
                )
            if final_line != -1 and final_line < init_line:
                raise ToolError(
                    f"无效的 'view_range': {view_range}。第二个元素 '{final_line}' 应该大于或等于第一个元素 '{init_line}'"
                )

            # Apply range
            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])

        # Format and return result
        return CLIResult(
            output=self._make_output(file_content, str(path), init_line=init_line)
        )

    async def str_replace(
        self,
        path: Path,
        old_str: str,
        new_str: str | None,
        operator: FileOperator = None,
    ) -> CLIResult:
        """执行字符串替换操作

        执行流程：
        1. 读取文件内容并展开制表符
        2. 检查旧字符串出现次数
        3. 唯一匹配时执行替换并保存历史
        4. 生成包含代码片段的成功响应

        Args:
            path: 目标文件路径
            old_str: 需要替换的原始字符串
            new_str: 替换后的新字符串(可选)

        Returns:
            CLIResult: 包含操作结果的响应对象

        Raises:
            ToolError: 当出现以下情况时抛出：
                - 未找到匹配内容
                - 发现多个匹配位置
        """
        # Read the file content
        file_content = (await operator.read_file(path)).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ""

        # Check if old_str is unique in the file
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise ToolError(
                f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
            )
        elif occurrences > 1:
            # Find line numbers of occurrences
            file_content_lines = file_content.split("\n")
            lines = [
                idx + 1
                for idx, line in enumerate(file_content_lines)
                if old_str in line
            ]
            raise ToolError(
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` "
                f"in lines {lines}. Please ensure it is unique"
            )

        # Replace old_str with new_str
        new_file_content = file_content.replace(old_str, new_str)

        # Write the new content to the file
        await operator.write_file(path, new_file_content)

        # Save the original content to history
        self._file_history[path].append(file_content)

        # Create a snippet of the edited section
        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

        # Prepare the success message
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet, f"a snippet of {path}", start_line + 1
        )
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return CLIResult(output=success_msg)

    async def insert(
        self,
        path: PathLike,
        insert_line: int,
        new_str: str,
        operator: FileOperator = None,
    ) -> CLIResult:
        """Insert text at a specific line in a file."""
        # Read and prepare content
        file_text = (await operator.read_file(path)).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        # Validate insert_line
        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}。它应该在文件的行数范围内: {[0, n_lines_file]}"
            )

        # Perform insertion
        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )

        # Create a snippet for preview
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        # Join lines and write to file
        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        await operator.write_file(path, new_file_text)
        self._file_history[path].append(file_text)

        # Prepare success message
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."

        return CLIResult(output=success_msg)

    async def undo_edit(
        self, path: PathLike, operator: FileOperator = None
    ) -> CLIResult:
        """Revert the last edit made to a file."""
        if not self._file_history[path]:
            raise ToolError(f"No edit history found for {path}.")

        old_text = self._file_history[path].pop()
        await operator.write_file(path, old_text)

        return CLIResult(
            output=f"Last edit to {path} undone successfully. {self._make_output(old_text, str(path))}"
        )

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ) -> str:
        """Format file content for display with line numbers."""
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()

        # Add line numbers to each line
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )

        return (
            f"Here's the result of running `cat -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
        )
