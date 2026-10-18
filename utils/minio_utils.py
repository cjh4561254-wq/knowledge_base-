# utils/minio_utils.py
"""
MinIO 对象存储封装层

职责：
1. 用 config.minio_config 建一个「全局复用」的 MinIO 客户端（懒加载 + 线程安全）。
2. 保证桶存在，并保证桶里的对象能被浏览器直接 GET（匿名可读策略）——
   Markdown 里的图片链接最终要被 chat 页面 <img src> 渲染出来。
3. 抹平 object key 的前导斜杠兼容性问题（MINIO_IMG_DIR=/upload-images 会拼出
   "/upload-images/xxx/1.jpg" 这种以 / 开头的 key）。
4. 统一上传/删除入口，失败时抛出 MinioError，绝不静默返回 None。

设计取舍（为什么是这样）：
- 采用「桶匿名可读 + 直链 URL」而不是 presigned URL：图片 URL 会被写进 Markdown
  并向量库/Mongo 持久化，presigned URL 带有效期（默认 7 天，最长 7 天），过期后
  链接变成死链；匿名可读策略下的 URL 永久有效、且形态最简单：
      http://{endpoint}/{bucket_name}/{object_key}
  presigned 能力仍然保留（get_presigned_url），仅用于私有桶场景的临时分享。
- get_minio_client() 初始化失败时抛异常而不是返回 None：
  原实现在模块 import 阶段吞掉异常，导致后续 minio_client.xxx 报
  AttributeError: 'NoneType'，排查成本极高；导入流程「存储层不可用」必须是显式失败，
  否则 Markdown 里会留下指向本地临时文件的死图片链接。
"""

import io
import json
import logging
import mimetypes
import os
import threading
from datetime import timedelta
from typing import Optional
from urllib.parse import quote

from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import S3Error

from config.minio_config import minio_config
from processor.import_processor.base import setup_logging
from processor.import_processor.exceptions import MinioError

setup_logging()  # 日志的全局配置 激活

logger = logging.getLogger("utils.minio")

# presigned URL 的默认有效期：MinIO/S3 签名链接上限 7 天，这里取保守值
DEFAULT_PRESIGNED_EXPIRES = timedelta(days=7)

# 全局单例状态：客户端 + 已校验过的 endpoint（懒加载，import 本模块不发网络请求）
_init_lock = threading.Lock()
_client: Optional["KbMinioClient"] = None
_current_endpoint: tuple[str, bool] | None = None


def _endpoint_info() -> tuple[str, bool]:
    """取已校验的 endpoint；未初始化时先初始化，避免拼接出错误 URL"""
    if _current_endpoint is None:
        get_minio_client()
    return _current_endpoint  # type: ignore[return-value]


# ==================== 1. 配置归一化 ====================

def normalize_endpoint(endpoint: str) -> tuple[str, bool]:
    """
    把 .env 里的 MINIO_ENDPOINT 归一化成 Minio() 需要的形式。

    为什么必须做：Minio 的第一个参数只接受 `host:port`，它自己会按 secure 拼协议。
    如果写成 `http://localhost:9100`，SDK 会把 "http" 当成主机名去做 DNS 解析，
    报 InvalidURL / NameResolutionError —— 这是 MinIO 封装层最高频的坑。

    :param endpoint: 原始配置值，容忍带 scheme、带尾部斜杠
    :return: (host:port, secure) —— scheme 为 https 时 secure 置 True，否则尊重原配置的 http
    """
    raw = (endpoint or "").strip()
    if not raw:
        raise MinioError(
            message="MINIO_ENDPOINT 未配置，请检查 .env（本环境应为 localhost:9100）",
            node_name="minio_utils",
        )

    secure = False
    lowered = raw.lower()
    if lowered.startswith("https://"):
        secure = True
        raw = raw[len("https://"):]
    elif lowered.startswith("http://"):
        secure = False
        raw = raw[len("http://"):]

    # 去掉可能误配的 path / 尾部斜杠，只留 host:port
    raw = raw.split("/", 1)[0].strip()
    if not raw:
        raise MinioError(
            message=f"MINIO_ENDPOINT 配置非法：{endpoint!r}",
            node_name="minio_utils",
        )
    return raw, secure


def normalize_object_name(name: str | None, strip_trailing_slash: bool = False) -> str | None:
    """
    归一化 object key。

    为什么必须做：.env 里 MINIO_IMG_DIR=/upload-images（以 / 开头），
    node_md_img 拼出来的 key 形如 "//upload-images/文档名/图片.jpg"，
    而 list_objects(prefix="/upload-images/...") 又匹配不到任何对象
    （实测：写入时 MinIO 服务端会吞掉前导斜杠，但列举时不会，读写两侧不对称，
    于是「清理旧图片」永远删不掉东西，重复导入会堆积历史对象）。
    这里统一在客户端侧去掉前导斜杠并合并重复斜杠，读写两边都落在同一个 key 上。

    :param name: 原始 key / prefix
    :param strip_trailing_slash: 列举单个对象时置 True；列举「目录前缀」时要保留尾斜杠
    """
    if name is None:
        return None
    normalized = "/".join(part for part in str(name).split("/") if part not in ("", "."))
    if strip_trailing_slash:
        normalized = normalized.rstrip("/")
    return normalized


def build_public_url(object_name: str) -> str:
    """
    生成浏览器可直接打开的永久直链（本项目主方案）。

    形态：http(s)://host:port/{bucket}/{object_key}
    与 node_md_img 里 f"http://{endpoint}/{bucket}/{object_name}" 的拼法保持一致，
    区别是这里的 key 已归一化（不会出现 bucket//upload-images 这种双斜杠）。
    """
    endpoint, secure = _endpoint_info()
    scheme = "https" if secure else "http"
    key = normalize_object_name(object_name, strip_trailing_slash=True)
    # 必须百分号编码：中文/空格对象名不编码时，S3 侧会直接返回 400（curl 实测），
    # 只有编码后的 key 才能稳定拿到 200 image/png。safe="/" 保留路径层级。
    quoted_key = quote(key, safe="/")
    return f"{scheme}://{endpoint}/{minio_config.bucket_name}/{quoted_key}"


def get_base_url() -> str:
    """MinIO 访问基地址（供日志/其它包拼接用）"""
    endpoint, secure = _endpoint_info()
    return f"{'https' if secure else 'http'}://{endpoint}"


# ==================== 2. 客户端：在官方 Minio 之上做 key 归一化 + 异常包装 ====================

class KbMinioClient(Minio):
    """
    项目定制 MinIO 客户端。

    只干两件事，不改变任何官方方法签名（调用方仍可按位置/关键字方式调用）：
    1. 传入的 object_name / prefix 一律归一化（去前导斜杠、合并重复斜杠）；
    2. 底层 S3Error 统一包成 MinioError，带上桶名和 key，方便定位。
    """

    @staticmethod
    def _wrap(action: str, bucket: str, key: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except S3Error as e:
            raise MinioError(
                message=f"MinIO {action}失败：bucket={bucket}, key={key!r}, 原因={e}",
                node_name="minio_utils",
                cause=e,
            ) from e

    def fput_object(self, bucket_name, object_name, file_path, *args, **kwargs):
        key = normalize_object_name(object_name, strip_trailing_slash=True)
        return self._wrap("fput_object", bucket_name, key,
                          super().fput_object, bucket_name, key, file_path, *args, **kwargs)

    def put_object(self, bucket_name, object_name, data, length, *args, **kwargs):
        key = normalize_object_name(object_name, strip_trailing_slash=True)
        return self._wrap("put_object", bucket_name, key,
                          super().put_object, bucket_name, key, data, length, *args, **kwargs)

    def get_object(self, bucket_name, object_name, *args, **kwargs):
        key = normalize_object_name(object_name, strip_trailing_slash=True)
        return self._wrap("get_object", bucket_name, key,
                          super().get_object, bucket_name, key, *args, **kwargs)

    def stat_object(self, bucket_name, object_name, *args, **kwargs):
        key = normalize_object_name(object_name, strip_trailing_slash=True)
        return self._wrap("stat_object", bucket_name, key,
                          super().stat_object, bucket_name, key, *args, **kwargs)

    def remove_object(self, bucket_name, object_name, *args, **kwargs):
        key = normalize_object_name(object_name, strip_trailing_slash=True)
        return self._wrap("remove_object", bucket_name, key,
                          super().remove_object, bucket_name, key, *args, **kwargs)

    def presigned_get_object(self, bucket_name, object_name, *args, **kwargs):
        key = normalize_object_name(object_name, strip_trailing_slash=True)
        return self._wrap("presigned_get_object", bucket_name, key,
                          super().presigned_get_object, bucket_name, key, *args, **kwargs)

    def presigned_put_object(self, bucket_name, object_name, *args, **kwargs):
        key = normalize_object_name(object_name, strip_trailing_slash=True)
        return self._wrap("presigned_put_object", bucket_name, key,
                          super().presigned_put_object, bucket_name, key, *args, **kwargs)

    def list_objects(self, bucket_name, prefix=None, *args, **kwargs):
        # 前缀目录列举：保留尾部斜杠，否则 "upload-images/doc" 会误匹配 "upload-images/doc-x/"
        normalized_prefix = normalize_object_name(prefix) if prefix else prefix
        try:
            return list(super().list_objects(bucket_name, normalized_prefix, *args, **kwargs))
        except S3Error as e:
            raise MinioError(
                message=f"MinIO list_objects失败：bucket={bucket_name}, prefix={normalized_prefix!r}, 原因={e}",
                node_name="minio_utils",
                cause=e,
            ) from e

    def remove_objects(self, bucket_name, delete_list, *args, **kwargs):
        objects: list[DeleteObject] = []
        for item in delete_list:
            if isinstance(item, DeleteObject):
                # minio 7.x 的 DeleteObject 只有 .name（不是 .object_name），
                # 之前按 object_name 取值会 AttributeError，导致「先删后传」直接失败
                raw_name = item.name
                version_id = item.version_id
            else:
                raw_name = item
                version_id = None
            objects.append(DeleteObject(
                normalize_object_name(raw_name, strip_trailing_slash=True),
                version_id,
            ))
        return self._wrap("remove_objects", bucket_name, "<batch>",
                          super().remove_objects, bucket_name, objects, *args, **kwargs)


# ==================== 3. 桶就绪：自动建桶 + 匿名可读策略 ====================

def _public_read_policy(bucket_name: str) -> dict:
    """
    只读策略：允许匿名 GetObject（浏览器打开图片必需），不含 ListBucket / 写权限。
    最小权限原则：不给匿名 Put/Delete，避免知识库被外部篡改。
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
            },
        ],
    }


def _policy_grants_anonymous_read(policy_text: str | None, bucket_name: str) -> bool:
    """判断现有策略是否已经允许匿名读，避免每次导入都重复 set_policy（幂等 + 少一次写）"""
    if not policy_text:
        return False
    try:
        policy = json.loads(policy_text) if isinstance(policy_text, str) else policy_text
    except (TypeError, ValueError):
        return False

    resource = f"arn:aws:s3:::{bucket_name}/*"
    for statement in policy.get("Statement", []) or []:
        if statement.get("Effect") != "Allow":
            continue
        actions = statement.get("Action")
        actions = actions if isinstance(actions, list) else [actions]
        if not any(str(a).lower() in ("s3:getobject", "s3:getobject*", "*") for a in actions):
            continue
        principals = statement.get("Principal")
        principal_values = (principals.get("AWS") if isinstance(principals, dict) else principals) or "*"
        principal_values = principal_values if isinstance(principal_values, list) else [principal_values]
        if "*" not in [str(v) for v in principal_values]:
            continue
        resources = statement.get("Resource")
        resources = resources if isinstance(resources, list) else [resources]
        if any(resource == str(r) or str(r).endswith("/*") for r in resources):
            return True
    return False


def ensure_bucket_ready(client: Minio) -> None:
    """
    保证桶存在且对象可被浏览器直接读取。

    幂等：桶已存在就不重复创建；策略已具备匿名读就不重复设置。
    """
    bucket_name = minio_config.bucket_name
    if not bucket_name:
        raise MinioError(
            message="MINIO_BUCKET_NAME 未配置（本环境应为 knowledge-base）",
            node_name="minio_utils",
        )

    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"MinIO 桶不存在，已创建：{bucket_name}")

        try:
            current_policy = client.get_bucket_policy(bucket_name)
        except S3Error as e:
            # 桶没策略时 MinIO 返回 NoSuchBucketPolicy，属于「还没设过」的正常状态
            if e.code != "NoSuchBucketPolicy":
                logger.warning(f"读取桶策略失败（将直接重设只读策略）：{e}")
            current_policy = None

        if not _policy_grants_anonymous_read(current_policy, bucket_name):
            client.set_bucket_policy(bucket_name, json.dumps(_public_read_policy(bucket_name)))
            logger.info(f"MinIO 桶匿名只读策略已设置：{bucket_name}（图片直链可被浏览器访问）")
    except MinioError:
        raise
    except S3Error as e:
        raise MinioError(
            message=f"MinIO 桶初始化失败：bucket={bucket_name}, 原因={e}",
            node_name="minio_utils",
            cause=e,
        ) from e


# ==================== 4. 懒加载单例 ====================

def get_minio_client() -> Minio:
    """
    获取已就绪的 MinIO 客户端（懒加载 + 线程安全 + 失败显式抛错）。

    为什么懒加载：原来在模块顶层建链，任何 `import utils.minio_utils` 都会发一次
    网络请求（bucket_exists + set_bucket_policy），MinIO 未就绪时整个 Web 服务都起不来，
    而且异常被吞掉后 minio_client 变成 None，报错信息惨不忍睹。

    :return: KbMinioClient（Minio 子类，签名与官方一致）
    :raises MinioError: 配置缺失 / 连接不上 / 建桶设策略失败
    """
    global _client, _current_endpoint

    if _client is not None:
        return _client

    with _init_lock:
        if _client is not None:
            return _client

        endpoint, secure = normalize_endpoint(minio_config.endpoint)

        # 防呆：本机 9000 端口是 Milvus 自带的内部对象存储（kb-milvus-minio），
        # 应用侧 MinIO 是 9100（kb-app-minio），两者串了会把图片写进 Milvus 的存储里。
        if endpoint.endswith(":9000"):
            logger.warning(
                "当前 MinIO endpoint 指向 9000 端口，本项目的 9000 是 Milvus 内部对象存储，"
                "应用侧 MinIO 应为 localhost:9100，请确认 MINIO_ENDPOINT 是否配错"
            )

        # region 不传时 SDK 走 GetBucketLocation 自动探测，一般没问题；
        # 多站点/离线环境可用环境变量 MINIO_REGION 显式指定（可选，未配置不影响）。
        client = KbMinioClient(
            endpoint=endpoint,  # 关键：host:port，不带 http:// 前缀
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=secure,  # 本机 http
            region=os.getenv("MINIO_REGION") or None,
        )

        ensure_bucket_ready(client)

        _current_endpoint = (endpoint, secure)
        _client = client
        logger.info(
            f"MinIO 客户端就绪：{get_base_url()}，桶：{minio_config.bucket_name}，"
            f"图片目录前缀：{normalize_object_name(minio_config.img_dir)}"
        )
        return _client


def get_public_base_url() -> str:
    """兼容 config_base.get_minio_base_url() 的语义：返回 http://host:port（无尾部斜杠）"""
    return get_base_url()


# ==================== 5. 上传 / 删除 / 临时授权 ====================

def upload_file(local_path: str, object_name: str, content_type: str | None = None,
                node_name: str = "minio_utils") -> str:
    """
    上传单个文件并返回浏览器可访问的永久直链。

    失败一定抛 MinioError —— 调用方（node_md_img）绝不能 catch 后返回 None，
    否则 Markdown 会保留指向本地临时目录的死图片链接，向量库存进去就是坏数据。

    幂等/覆盖：同名 object_key 直接覆盖（last-write-wins，桶未开版本控制），
    所以同一文档重复导入产生的 URL 不变，不会产生副本。
    """
    if not os.path.exists(local_path):
        raise MinioError(
            message=f"待上传文件不存在：{local_path}",
            node_name=node_name,
        )

    client = get_minio_client()
    key = normalize_object_name(object_name, strip_trailing_slash=True)
    mime = content_type or mimetypes.guess_type(local_path)[0] or "application/octet-stream"

    client.fput_object(
        bucket_name=minio_config.bucket_name,
        object_name=key,
        file_path=local_path,
        content_type=mime,
    )
    url = build_public_url(key)
    logger.info(f"MinIO 上传成功：{key} -> {url}")
    return url


def delete_prefix(prefix: str, node_name: str = "minio_utils") -> int:
    """
    清空某个前缀（重新导入前的「先删后传」）。

    :return: 实际删除的对象数
    """
    client = get_minio_client()
    normalized = normalize_object_name(prefix)
    if not normalized:
        return 0

    to_delete = [DeleteObject(obj.object_name) for obj in
                 client.list_objects(minio_config.bucket_name, normalized, recursive=True)]
    if not to_delete:
        return 0

    errors = list(client.remove_objects(minio_config.bucket_name, to_delete)) or []
    for error in errors:
        logger.error(f"MinIO 删除对象失败：{error}")
    if errors:
        raise MinioError(
            message=f"MinIO 批量删除存在 {len(errors)} 个错误：{errors[0]}",
            node_name=node_name,
        )
    logger.info(f"MinIO 已清理前缀 {normalized}，共 {len(to_delete)} 个对象")
    return len(to_delete)


def get_presigned_url(object_name: str,
                      expires: timedelta = DEFAULT_PRESIGNED_EXPIRES,
                      node_name: str = "minio_utils") -> str:
    """
    临时授权链接（备用方案）。

    仅适用于桶不开放匿名读的场景，例如导出交付物、临时对外分享。
    不要把它写进 Markdown：有效期最长 7 天，过期后知识库里的图片全部变死链。
    """
    client = get_minio_client()
    return client.presigned_get_object(
        bucket_name=minio_config.bucket_name,
        object_name=normalize_object_name(object_name, strip_trailing_slash=True),
        expires=expires,
    )


def upload_bytes(data: bytes, object_name: str, content_type: str = "application/octet-stream",
                 node_name: str = "minio_utils") -> str:
    """上传内存中的字节（自测/生成缩略图用），返回永久直链；同名覆盖"""
    client = get_minio_client()
    key = normalize_object_name(object_name, strip_trailing_slash=True)
    client.put_object(minio_config.bucket_name, key, io.BytesIO(data), len(data),
                      content_type=content_type)
    return build_public_url(key)


def list_objects(prefix: str = "", recursive: bool = True) -> list[str]:
    """列出前缀下的对象 key（已归一化，不含前导斜杠）"""
    client = get_minio_client()
    return [obj.object_name for obj in
            client.list_objects(minio_config.bucket_name, normalize_object_name(prefix), recursive)]


# ==================== 6. 兼容旧的模块级属性写法 ====================

def __getattr__(name: str):
    """
    PEP 562：兼容历史代码 `from utils.minio_utils import minio_client`。
    访问时才真正建链，避免 import 阶段发网络请求。
    """
    if name == "minio_client":
        return get_minio_client()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    # 只读自检：不打真实图片目录，只做 连通性 + 桶 + 策略 + 列举
    client = get_minio_client()
    print("endpoint          :", get_base_url())
    print("bucket            :", minio_config.bucket_name)
    print("bucket_exists     :", client.bucket_exists(minio_config.bucket_name))
    print("img_dir prefix    :", normalize_object_name(minio_config.img_dir))
    print("sample public url :", build_public_url(f"{normalize_object_name(minio_config.img_dir)}/demo/a.jpg"))
    print("objects in bucket :", len(list_objects("")) )
