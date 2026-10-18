# knowledge/processor/import_processor/exceptions.py

"""
导入流程自定义异常类

统一错误处理，提供更清晰的错误信息

异常层级（与 day13/03 骨架代码文档一致）：

    ImportProcessError (基础异常)
    ├── ConfigurationError        # 配置错误
    ├── StateFieldError           # 状态字段缺失/类型不符
    ├── FileProcessingError       # 文件处理错误
    │   ├── PdfConversionError    # PDF 转换错误
    │   └── ImageProcessingError  # 图片处理错误
    ├── DocumentSplitError        # 文档切分错误
    ├── EmbeddingError            # 向量化错误
    ├── LLMError                  # LLM 调用错误
    ├── StorageError              # 存储错误
    │   ├── MilvusError           # Milvus 存储错误
    │   └── MinioError            # MinIO 存储错误
    └── ValidationError           # 数据验证错误

对外契约：
  1. base.BaseNode.__call__ 只保证抛 ImportProcessError 家族；
     子类型原样透传（不再二次包装），所以 main_graph / web 层可以按类型分层捕获，
     也可以用「except ImportProcessError」一把兜住整条导入流水线。
  2. 所有异常都能通过 str(e) / user_message 拿到一句中文可读文案，
     通过 to_dict() 拿到可 JSON 化的错误结构（FastAPI 直接返回给前端）。
"""

class ImportProcessError(Exception):
    """导入流程基础异常"""

    def __init__(self, message: str, node_name: str = "", cause: Exception = None):
        self.node_name = node_name # 发生错误的流程节点名称
        self.cause = cause  # 导致该错误的底层原始异常
        super().__init__(message)
        # 显式挂异常链：raise 处没写 from 时，traceback 里仍能看到底层原因，
        # 便于把「Milvus 连接被拒」这类真实根因和上层业务提示一起打到日志里。
        if cause is not None and self.__cause__ is None:
            self.__cause__ = cause

    def __str__(self):
        parts = []
        if self.node_name:
            parts.append(f"[{self.node_name}]")
        parts.append(super().__str__())
        if self.cause:
            parts.append(f"(原因: {self.cause})")
        return " ".join(parts)

    @property
    def user_message(self) -> str:
        """面向用户的可读文案（web 层可直接展示，无需再拼装）。"""
        return str(self)

    def to_dict(self) -> dict:
        """可序列化的错误结构，供接口返回 / 日志落库。"""
        return {
            "error_type": type(self).__name__,
            "node_name": self.node_name,
            "message": super().__str__(),
            "cause": str(self.cause) if self.cause else "",
        }



class StateFieldError(ImportProcessError):
    """状态字段错误。

    从 state 中获取必需字段缺失、为空或类型不符时抛出。

    Attributes:
        field_name: 缺失或无效的字段名称。
        expected_type: 期望的字段类型（可选）。
    """

    def __init__(
            self,
            node_name: str = "",
            field_name: str = "",
            expected_type: type = None,
            message: str = "",
            cause: Exception = None,
    ):
        self.field_name = field_name
        self.expected_type = expected_type
        if not message:
            message = f"状态字段 '{field_name}' 缺失或无效"
            if expected_type:
                message += f"，期望类型: {expected_type.__name__}"
        super().__init__(message, node_name=node_name, cause=cause)


class ConfigurationError(ImportProcessError):
    """配置错误：环境变量缺失或配置值无效"""
    pass


class FileProcessingError(ImportProcessError):
    """文件处理错误：文件不存在、格式错误、读写失败"""
    pass


class PdfConversionError(FileProcessingError):
    """PDF 转换错误：MinerU 转换失败"""
    pass


class ImageProcessingError(FileProcessingError):
    """图片处理错误：图片总结、上传失败"""
    pass


class DocumentSplitError(ImportProcessError):
    """文档切分错误：切分逻辑异常"""
    pass


class EmbeddingError(ImportProcessError):
    """向量化错误：模型调用失败、向量生成异常"""
    pass


class LLMError(ImportProcessError):
    """LLM 调用错误：API 调用失败、响应解析失败"""
    pass


class StorageError(ImportProcessError):
    """存储错误：数据库操作失败"""
    pass


class MilvusError(StorageError):
    """Milvus 存储错误"""
    pass


class MinioError(StorageError):
    """MinIO 存储错误"""
    pass


class ValidationError(ImportProcessError):
    """数据验证错误：输入数据不符合预期"""
    pass



