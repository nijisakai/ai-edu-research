#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any
import requests
from openai import OpenAI
from tqdm import tqdm
import io
import base64

try:
    import docx
    from docx import Document  # 确保 Document 类正确导入
except ImportError as e:
    print(f"请安装 python-docx 库: pip install python-docx")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 设置全局变量
lock = threading.Lock()
processed_count = 0
backup_lock = threading.Lock()
processed_results = []
BACKUP_INTERVAL = 10  # 每处理10个文件进行一次备份

# 定义期望的输出格式模板 (精简，示例数据将会扩展)
OUTPUT_TEMPLATE = {
    "id": "",
    "title": "",
    "abstract": "",
    "authors": [],
    "citation_count": 0  # 新增: 引用次数
}

# 知识图谱实体示例 (简化)
KG_ENTITY_PROMPT = """
提取以下信息，并按照JSON格式返回，无需任何解释性文字。务必使用双引号。
{
    "researchers": [
        {"name": "研究者姓名", "orcid_id": "ORCID", "affiliation_history": [], "research_interests": []}
    ],
    "concepts": [
        {"name": "概念名称", "definition": "概念定义", "related_concepts": []}
    ],
    "theories": [
        {"name": "理论名称", "proponents": [], "criticisms": []}
    ],
    "methodologies": [
        {"name": "方法论名称", "method_type": "", "tools": []}
    ],
    "papers": [
        {"title": "论文标题", "doi": "DOI", "keywords": [], "publication_date": ""}
    ]
}
"""

def read_docx(file_path: Path) -> str:
    """读取docx文件内容"""
    try:
        doc = Document(file_path)  # 使用 Document 类
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return '\n'.join(full_text)
    except Exception as e:
        logger.error(f"读取docx文件失败: {str(e)}")
        return ""

class PDFProcessor:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "qwen-long",
        max_workers: int = 5,
        backup_interval: int = BACKUP_INTERVAL
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.backup_interval = backup_interval
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def process_file(self, file_path: Path, output_dir: Path) -> Dict:
        """处理单个文件并返回结果"""
        global processed_count, processed_results
        logger.info(f"开始处理文件: {file_path}")

        # 获取完整文件名（包含扩展名）作为id
        file_id = file_path.name

        try:
            # 根据文件类型读取内容
            file_suffix = file_path.suffix.lower()
            if file_suffix == ".pdf":
                with open(file_path, "rb") as f:
                    file_object = self.client.files.create(
                        file=f,
                        purpose="file-extract"
                    )
            elif file_suffix == ".docx":
                text_content = read_docx(file_path)
                # 将文本内容编码为 bytes
                byte_content = text_content.encode('utf-8')  # 使用 UTF-8 编码，或其他合适的编码
                # 创建一个内存文件对象，以便传递给OpenAI API
                file_object = self.client.files.create(
                    file=io.BytesIO(byte_content),  # 使用 io.BytesIO 将字节数据转换为文件对象
                    purpose="file-extract"
                )
            else:
                raise ValueError(f"不支持的文件类型: {file_path.suffix}")

            api_file_id = file_object.id

            # 使用模型分析文档，要求严格按照以下JSON格式输出结果
            prompt = f"""
            请分析这篇学术论文，首先提取基本信息，然后提取知识图谱相关实体。务必遵守以下要求:

            1.  基本信息提取，严格按照以下JSON格式输出结果：
            {{
                "id": "{file_id}",
                "title": "论文标题",
                "abstract": "论文摘要",
                "authors": ["作者1", "作者2"],
                "citation_count": 0  # 估算论文的引用次数 (整数).  如果无法确定，请使用0
            }}

            2.  知识图谱实体提取，严格按照以下JSON格式输出结果:
            {KG_ENTITY_PROMPT}

            只输出JSON格式内容，不要添加任何额外文字、代码块标记或说明。
            如果无法提取某个字段，使用null值。 确保JSON格式正确。
            """

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in academic paper analysis and knowledge graph extraction."},
                    {"role": "system", "content": f"fileid://{api_file_id}"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )

            # 提取响应文本
            response_text = completion.choices[0].message.content.strip()

            # 解析JSON (处理可能存在的代码块)
            try:
                if "```json" in response_text:
                    json_content = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_content = response_text.split("```")[1].strip()
                else:
                    json_content = response_text

                # 分割成两部分
                parts = json_content.split('}\n{') # 假设基本信息和KG信息之间用这个分割
                if len(parts) == 2: # 如果成功分割
                    basic_info_str = parts[0] + '}'  # 补全
                    kg_info_str = '{' + parts[1]    # 补全
                else: # 如果分割失败，全部作为基本信息处理（或报错）
                    basic_info_str = response_text  # 如果无法分割，则整个响应作为基本信息
                    kg_info_str = "{}"  # 空的知识图谱

                basic_info = json.loads(basic_info_str) if basic_info_str else {}
                kg_info = json.loads(kg_info_str) if kg_info_str else {}

                # 合并结果
                paper_info = {**basic_info, "kg_entities": kg_info}

                # 确保输出格式符合要求 (只检查基本信息部分)
                for key in OUTPUT_TEMPLATE:
                    if key not in paper_info:
                        paper_info[key] = OUTPUT_TEMPLATE[key]

                # 强制设置正确的ID为文件名
                paper_info["id"] = file_id

            except json.JSONDecodeError as je:
                logger.error(f"解析JSON失败: {str(je)}, 原始响应: {response_text[:200]}...")
                paper_info = {**OUTPUT_TEMPLATE, "id": file_id, "error": "JSON解析失败", "file_path": str(file_path)}

            # 更新处理计数和保存结果
            with lock:
                processed_results.append(paper_info)
                processed_count += 1
                # 检查是否需要备份
                if processed_count % self.backup_interval == 0:
                    self._backup_results(output_dir, processed_count)
            logger.info(f"文件处理完成: {file_path}")
            return paper_info

        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            error_result = {**OUTPUT_TEMPLATE, "id": file_id, "error": str(e), "file_path": str(file_path)}
            return error_result

    def _backup_results(self, output_dir: Path, count: int) -> None:
        """创建增量备份 - 使用JSONL格式"""
        with backup_lock:
            backup_file = output_dir / f"papers_backup_{count}.json"
            logger.info(f"创建增量备份: {backup_file}")
            with open(backup_file, "w", encoding="utf-8") as f:
                for paper in processed_results:
                    f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    def process_batch(self, input_dir: Path, output_dir: Path, file_extensions: List[str] = [".pdf"]) -> List[Dict]:
        """批量处理文件夹中的所有文件"""
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        # 获取所有文件
        files_to_process = []
        for ext in file_extensions:
            files_to_process.extend(list(input_dir.glob(f"*{ext}")))
        if not files_to_process:
            logger.warning(f"在 {input_dir} 中没有找到符合条件的文件")
            return []
        logger.info(f"找到 {len(files_to_process)} 个文件需要处理")
        global processed_results
        processed_results = []  # 重置结果列表

        # 使用线程池并发处理文件
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_file, file_path, output_dir) for file_path in files_to_process]
            # 使用tqdm显示进度
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
                try:
                    future.result()  # 获取结果并处理潜在的异常
                except Exception as e:
                    logger.error(f"在future中捕获异常: {e}") # 在这里处理future中的异常

        # 完成后保存全部结果为JSONL格式
        final_output = output_dir / "all_papers.json"
        with open(final_output, "w", encoding="utf-8") as f:
            for paper in processed_results:
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")
        logger.info(f"所有文件处理完成，最终结果保存至 {final_output}")
        return processed_results

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量处理文件并输出分析结果为JSONL格式")
    parser.add_argument("--input_dir", type=str, default="./", help="输入文件夹路径")
    parser.add_argument("--output_dir", type=str, default="./", help="输出文件夹路径")
    parser.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL"),
                       help="OpenAI API兼容接口的基础URL")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY"),
                       help="OpenAI API密钥")
    parser.add_argument("--model", type=str, default="qwen-long",
                       help="要使用的模型名称")
    parser.add_argument("--max_workers", type=int, default=5,
                       help="最大并行工作线程数")
    parser.add_argument("--extensions", type=str, default=".pdf,.docx",
                       help="要处理的文件扩展名，以逗号分隔")
    args = parser.parse_args()
    # 检查必要参数
    if not args.base_url:
        raise ValueError("必须提供base_url参数或设置OPENAI_BASE_URL环境变量")
    if not args.api_key:
        raise ValueError("必须提供api_key参数或设置OPENAI_API_KEY环境变量")
    return args

def main():
    """主函数"""
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    extensions = [ext.strip() for ext in args.extensions.split(",")]
    logger.info(f"启动文件处理器，输入目录: {input_dir}, 输出目录: {output_dir}")
    logger.info(f"使用模型: {args.model}, 最大工作线程: {args.max_workers}")
    processor = PDFProcessor(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_workers=args.max_workers
    )
    processor.process_batch(input_dir, output_dir, extensions)

if __name__ == "__main__":
    main()