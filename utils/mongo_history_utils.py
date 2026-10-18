# utils/mongo_history_utils.py

# 导入系统模块：用于读取环境变量
import os
# 导入日志模块：用于记录程序运行日志（成功/失败/错误信息）

from tool.logger import logger
# 导入类型注解模块：用于函数参数/返回值的类型提示，提升代码可读性和规范性
from typing import List, Dict, Any, Optional
# 导入时间模块：用于生成时间戳，记录对话的创建时间
from datetime import datetime
# 导入pymongo核心模块：MongoDB原生Python驱动，实现数据库连接和操作
# ASCENDING：表示升序排序，用于MongoDB索引和查询排序
from pymongo import MongoClient, ASCENDING, DESCENDING
# 导入bson的ObjectId：MongoDB默认的主键类型，用于唯一标识文档
from bson import ObjectId
# 导入dotenv模块：用于从.env文件加载环境变量，避免硬编码敏感配置（如MongoDB连接地址）
from dotenv import load_dotenv

# 加载.env文件中的环境变量，使os.getenv能读取到配置
load_dotenv(override=True)

# 链接MongoDB工具类
class HistoryMongoTool:
    """
    MongoDB 历史对话记录读写工具类 (基于原生 PyMongo 实现)
    核心功能：封装MongoDB的连接、集合初始化、索引创建，为上层提供统一的数据库操作入口
    扩展功能：支持与LangChain消息对象的格式转换（原代码预留能力）
    """

    # MongoDB 不可达时的"快速失败"超时（毫秒）
    # 为什么：pymongo 默认 serverSelectionTimeoutMS=30000，连接不上时每个操作都会把
    # 请求线程卡死 30 秒；这里压到 5 秒，保证 Mongo 故障时接口快速报错而不是整进程假死
    _SERVER_SELECTION_TIMEOUT_MS = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "5000"))

    def __init__(self):
        """
        类初始化方法：完成MongoDB的连接、数据库/集合获取、索引创建
        注意：pymongo 的 MongoClient 是惰性连接，构造时不会真正拨号；
        真正的网络动作发生在第一次读写操作（这里是 create_index），
        因此索引创建失败只降级为告警，不让初始化抛异常（见下方说明）
        """
        # 从环境变量读取MongoDB连接地址（敏感配置，不硬编码）
        # 兜底默认值：即使 .env 缺失/加载顺序异常，也能连本机容器库，避免 None 导致崩溃
        self.mongo_url = os.getenv("MONGO_URL") or "mongodb://localhost:27017"
        # 从环境变量读取要使用的数据库名称（未配置时回退课程约定的 kb001）
        self.db_name = os.getenv("MONGO_DB_NAME") or "kb001"

        try:
            # 创建MongoDB客户端实例（惰性，不触发网络）
            self.client = MongoClient(
                self.mongo_url,
                serverSelectionTimeoutMS=self._SERVER_SELECTION_TIMEOUT_MS,
                connectTimeoutMS=self._SERVER_SELECTION_TIMEOUT_MS,
            )
            # 获取指定名称的数据库对象（pymongo 中数据库/集合均为访问时引用，不要求预先存在）
            self.db = self.client[self.db_name]
            # 获取对话记录的集合（相当于关系型数据库的表），集合名：chat_message
            # 集合不存在时无需手动建：首次 insert_one 会自动创建集合，保证"空库首启"可用
            self.chat_message = self.db["chat_message"]

            # 为chat_message集合创建复合索引，提升查询性能
            # 索引规则：session_id升序 + ts降序，适配"按会话查最新记录"的核心查询场景
            # create_index自带幂等性：索引已存在时不会重复创建，无需额外判断
            # 为什么包 try/except：这是初始化路径里唯一会真正联网的操作，
            # Mongo 暂时不可达时（5 秒超时）不应让整个工具类构造失败——
            # 索引可以在下次进程/实例重建时补上，读写函数内部各自还有异常兜底
            try:
                self.chat_message.create_index(
                    [("session_id", ASCENDING), ("ts", DESCENDING)],
                    name="idx_session_id_ts",
                )
                # 记录成功日志，确认数据库连接和索引初始化完成
                logger.info(f"Successfully connected to MongoDB: {self.db_name}")
            except Exception as e:
                logger.warning(
                    f"Connected lazily to MongoDB ({self.db_name}), but index creation failed: {e}. "
                    f"Chat history reads/writes may be slow or unavailable until Mongo recovers."
                )
        except Exception as e:
            # 捕获所有初始化异常，记录详细错误日志
            logger.error(f"Failed to connect to MongoDB: {e}")
            # 重新抛出异常，让调用方感知初始化失败，避免使用未初始化的实例
            raise


# 定义全局变量：存储HistoryMongoTool的单例实例
# 为什么初始为 None 而不是在模块加载时直接实例化：
# 顶层 new HistoryMongoTool() 会在 import 阶段就触发 create_index 网络操作，
# Mongo 未就绪时整个进程要么卡死（超时）、要么 import 直接抛异常导致服务起不来；
# 改成真正的懒加载：首次调用业务函数时才建立连接，import 永远轻量无副作用
_history_mongo_tool: Optional["HistoryMongoTool"] = None


def get_history_mongo_tool() -> HistoryMongoTool:
    """
    获取HistoryMongoTool的单例实例（懒加载模式）
    核心逻辑：全局实例为空时创建，不为空时直接返回，保证整个程序只有一个数据库连接实例
    :return: HistoryMongoTool的单例实例
    """
    # 声明使用全局变量，避免函数内视为局部变量
    global _history_mongo_tool
    # 懒加载：仅当全局实例为空时，才创建新的实例
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    # 返回单例实例
    return _history_mongo_tool


def save_chat_message(
        session_id: str,role: str,text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        image_urls: List[str] = None,
        message_id: str = None,
        sources: List[Dict[str, Any]] = None
) -> str:
    """
    写入/更新单条会话记录到MongoDB
    支持两种模式：无message_id时新增记录，有message_id时更新已有记录
    :param session_id: 会话唯一标识，关联对话所属的会话
    :param role: 消息角色，固定值：user（用户）/assistant（助手）
    :param text: 对话核心内容，用户的提问或助手的回答
    :param rewritten_query: 重写后的查询语句（可选，用于检索增强等场景，默认空字符串）
    :param item_names: 关联的商品名称列表（可选，支持多商品，默认None）
    :param image_urls: 关联的图片URL列表（可选，默认None）
    :param message_id: 记录主键ID（可选，有值则更新，无值则新增）
    :param sources: 关联的引用来源列表（可选，默认None）。元素为
                    node_answer_output.build_sources 的产物
                    {"index","ref","chunk_id","source","title","url","score"}，
                    index 与答案文本里的 [doc:N] 编号同源。
                    放在参数列表末尾且带默认值：所有现存调用方（用户消息侧的
                    node_item_name_confirm、非引用场景）不改动即可正常工作。
    :return: 插入/更新的记录唯一标识（新增返回ObjectId字符串，更新返回传入的message_id）
    """
    # 生成当前时间的时间戳（秒级），用于记录消息的创建时间，后续用于排序和查询
    ts = datetime.now().timestamp()

    # 构造要插入/更新的文档数据（MongoDB的基本数据单元是文档，类似Python字典）
    # item_names/image_urls/sources 为 None 时统一存空列表：保证库里字段类型恒为数组，
    # 前端 /history 接口拿到 null 还要额外判空，入库时就规范化
    # sources 是纯增量字段：历史老记录没有这个键，读取侧（get_recent_messages 的调用方
    # web/api/query_service.py:/history）已用 r.get("sources") or [] 兜底，因此无需数据迁移
    document = {
        "session_id": session_id,  # 会话ID，关联维度
        "role": role,  # 消息角色
        "text": text,  # 消息内容
        "rewritten_query": rewritten_query or "",  # 问题优化后的改写，空值处理为空字符串
        "item_names": item_names or [],  # 关联商品名称列表
        "image_urls": image_urls or [],  # 关联图片URL列表
        "sources": sources or [],  # 关联引用来源列表（与 [doc:N] 编号同源，仅助手消息有值）
        "ts": ts  # 时间戳，排序和时间筛选维度
    }

    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()
    # 判断是否传入主键ID，区分更新/新增逻辑
    if message_id:# 更新
        # 有message_id：执行更新操作（根据主键更新）
        result = mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},  # 更新条件：主键匹配（需将字符串转为ObjectId类型）
            {"$set": document}  # 更新操作：$set表示只更新指定字段，保留其他字段
        )
        # 更新操作返回传入的message_id作为标识
        return message_id
    else:
        # 无message_id：执行新增操作
        result = mongo_tool.chat_message.insert_one(document)
        # 新增操作返回插入的ObjectId并转为字符串，便于上层使用（避免直接返回ObjectId对象）
        return str(result.inserted_id)# message_id 唯一的一条记录


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """
    批量更新历史会话记录的关联商品名称
    :param ids: 要更新的记录主键ID列表（字符串类型）
    :param item_names: 要设置的新商品名称列表
    :return: 实际更新的文档数量，更新失败返回0
    """
    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()
    try:
        # 将字符串类型的主键列表转为MongoDB的ObjectId类型（数据库中主键是ObjectId类型）
        object_ids = [ObjectId(i) for i in ids]
        # 执行批量更新操作
        result = mongo_tool.chat_message.update_many(
            # 更新条件：复合条件，同时满足
            {
                "_id": {"$in": object_ids}  # 主键在指定的ID列表中（批量筛选）
            },
            {"$set": {"item_names": item_names}}  # 更新操作：设置新的商品名称列表
        )
        # 记录更新成功日志，包含更新数量和新的商品名称
        logger.info(f"Updated {result.modified_count} records to item_names: {item_names}")
        # 返回实际更新的数量（modified_count：真正被修改的文档数，区别于matched_count）
        return result.modified_count
    except Exception as e:
        # 捕获批量更新异常，记录错误日志
        logger.error(f"Error updating history item_names: {e}")
        # 异常时返回0，标识更新失败
        return 0

# 获取最近的消息
def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    查询指定会话的最近N条对话记录，返回原始字典格式
    结果按时间正序排列，可直接喂给LLM作为上下文
    :param session_id: 会话唯一标识，用于筛选指定会话的记录
    :param limit: 条数限制，默认返回最近10条
    :return: 对话记录列表（字典格式），查询失败返回空列表
    """
    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()# 拿到链接mongodb的客户端对象
    try:
        # 构造查询条件：仅查询指定session_id的记录
        query = {"session_id": session_id}# 你要查的那个会话

        # 先按 ts 倒序取最近 limit 条，再反转回正序。
        # 若直接 ASCENDING + limit，消息数超过 limit 时会拿到「最旧的 N 条」，
        # 喂给 LLM 的上下文就永远是开场白而不是最新对话。
        cursor = mongo_tool.chat_message.find(query).sort("ts", DESCENDING).limit(limit)
        # 将游标转为列表，触发实际数据库查询，获取所有符合条件的文档
        messages = list(cursor)
        messages.reverse()
        # 返回查询结果列表
        # 将每条文档的 _id 转成字符串，保证 JSON 可序列化
        for msg in messages:
            if "_id" in msg:
                msg["_id"] = str(msg["_id"])
        return messages
    except Exception as e:
        # 捕获查询异常，记录错误日志
        logger.error(f"Error getting recent messages: {e}")
        # 异常时返回空列表，避免上层处理None报错
        return []


def clear_history(session_id: str) -> int:
    """
    清空指定会话的所有历史对话记录
    :param session_id: 会话唯一标识，用于筛选要删除的记录
    :return: 实际删除的文档数量，删除失败返回0
    """
    # 获取全局的HistoryMongoTool实例，使用单例模式避免重复创建数据库连接
    mongo_tool = get_history_mongo_tool()
    try:
        # 执行批量删除操作：删除所有session_id匹配的文档
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        # 记录删除成功日志，包含删除数量和会话ID，便于问题排查
        logger.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        # 返回实际删除的数量（delete_many的返回对象包含deleted_count属性）
        return result.deleted_count
    except Exception as e:
        # 捕获删除异常，记录错误日志，包含会话ID
        logger.error(f"Error clearing history for session {session_id}: {e}")
        # 异常时返回0，标识删除失败
        return 0


if __name__ == "__main__":
    # # 测试代码
    # save_chat_message(
    #     "test_session_id_001", "user", "请问这个怎么用？", "请问HAK180烫金机如何使用？"
    # )
    clear_history("test_session_002")
    # 测试会话，用于确认商品名称是否能正确的提取
    # sid = "test_session_002"
    # # 1. 写入用户消息
    # save_chat_message(sid, "user", "你好，有烫金机吗？")
    # # 2. 写入助手回复
    # save_chat_message(sid, "assistant", "你好！请问你想询问哪个型号？")
    # # 3. 写入带关联商品的用户消息
    # save_chat_message(sid, "user", "brother的HAK180烫金机")
    # save_chat_message(sid, "assistant", "有的")


