import logging

# 1. 创建 Logger（记录器）
logger = logging.getLogger("file_logger")
logger.setLevel(logging.INFO)

# 2. 创建 FileHandler（文件处理器），指定文件名和编码（防止中文乱码）
file_handler = logging.FileHandler("app.log", encoding="utf-8")

# 3. 创建 Formatter（干净的纯文本格式器，不带任何颜色代码）
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 4. 组装
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


if __name__ == "__main__":
    # 5. 运行后，控制台没有任何输出，数据直接进文件
    logger.info("系统启动成功，日志已安全落盘")
    logger.error("由于网络波动，Milvus 写入数据重试了一次")