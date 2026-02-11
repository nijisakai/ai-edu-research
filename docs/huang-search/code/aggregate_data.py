# aggregate_data.py
import json
import logging
import argparse
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import networkx as nx
import io
import base64
import matplotlib.font_manager as fm
import sys
import os
# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s',
    handlers=[logging.FileHandler("aggregate_data.log", encoding='utf-8'),
              logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 配置中文字体：自动检测常见中文字体并应用到 matplotlib
def configure_chinese_font(preferred_fonts=None):
    """尝试在系统中找到一个可用的中文字体并设置为 matplotlib 的默认字体。
    preferred_fonts: 按优先级排列的字体名称列表。
    """
    if preferred_fonts is None:
        preferred_fonts = [
            "Noto Sans CJK SC",
            "NotoSansCJKsc",
            "Microsoft YaHei",
            "MicrosoftYaHei",
            "PingFang SC",
            "SimHei",
            "SimSun",
            "Heiti SC",
            "STHeiti",
            "Arial Unicode MS",
        ]
    try:
        available_names = {f.name for f in fm.fontManager.ttflist}
        chosen_font_path = None
        chosen_font_name = None
        for name in preferred_fonts:
            if name in available_names:
                try:
                    chosen_font_path = fm.findfont(name)
                    chosen_font_name = name
                    plt.rcParams['font.family'] = name
                    logger.info(f"配置中文字体: 使用 '{name}'")
                    break
                except Exception:
                    # fallback to next
                    chosen_font_path = None
                    chosen_font_name = None
        if chosen_font_path is None:
            # 若未在首选名单中找到，尝试通过关键字搜索候选字体
            for f in fm.fontManager.ttflist:
                lname = f.name.lower()
                if any(k in lname for k in ("noto", "cjk", "pingfang", "heiti", "sim", "msyh", "song")):
                    chosen_font_path = f.fname if hasattr(f, 'fname') else fm.findfont(f.name)
                    chosen_font_name = f.name
                    plt.rcParams['font.family'] = f.name
                    logger.info(f"配置中文字体: 检测并使用 '{f.name}'")
                    break
        if chosen_font_path:
            try:
                font_prop = fm.FontProperties(fname=chosen_font_path)
                return font_prop
            except Exception:
                # As a fallback, return a FontProperties with family name
                try:
                    return fm.FontProperties(family=chosen_font_name) if chosen_font_name else None
                except Exception:
                    return None
        else:
            logger.warning(
                "未检测到已知中文字体。图表中的中文可能无法正确显示。建议安装 'Noto Sans CJK' 或 'SimHei' 等字体。"
            )
            return None
    except Exception as e:
        logger.warning(f"配置中文字体时出错: {e}")
        return None


# 立即应用中文字体配置并保留 FontProperties 以供显式使用
font_prop = configure_chinese_font()

# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 如果你想明确指定某个字体文件路径，可以使用下面的方式：
# font_path = '/path/to/your/SimHei.ttf'  # 替换为你的字体文件路径
# font_prop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()

# --- Helper Functions for Visualization ---
def embed_chart_in_html(chart_buffer: io.BytesIO, alt_text: str = "Chart"):
    """Encodes chart to base64 and returns base64 string."""
    chart_buffer.seek(0)  # Rewind to the beginning of the buffer
    image_data = chart_buffer.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    #image_html = f'<img src="data:image/png;base64,{image_base64}" alt="{alt_text}" style="max-width:100%;">'
    return image_base64 # 返回base64编码

def generate_concept_cooccurrence_chart(concept_co_occurrence: Counter, top_n: int = 10):
    """
    Generates a horizontal bar chart of the top N concept co-occurrences.
    Returns base64 string
    """
    top_concepts = concept_co_occurrence.most_common(top_n)
    if not top_concepts:
        return ""

    try:
        concept_pairs, counts = zip(*top_concepts)
        #concept_labels = [f"{pair[0]} - {pair[1]}" for pair in concept_pairs]  # Format labels
        # 过滤concepts里的none
        concept_labels = [f"{pair[0]} - {pair[1]}" for pair in concept_pairs if pair[0] and pair[1]]  # Filter None values
        counts = counts[:len(concept_labels)] # 确保counts 和 concept_labels 长度一致

        plt.figure(figsize=(10, 6))  # Adjust figure size
        plt.barh(concept_labels, counts, color="#66b3ff")  # Use a softer color

        # Use explicit FontProperties when available to ensure Chinese renders correctly
        if 'font_prop' in globals() and font_prop is not None:
            plt.xlabel("Co-occurrence Count", fontsize=12, fontproperties=font_prop)
            plt.ylabel("Concept Pairs", fontsize=12, fontproperties=font_prop)
            plt.title(f"Top {top_n} Concept Co-occurrences", fontsize=14, fontproperties=font_prop)
            try:
                plt.yticks(fontproperties=font_prop)
                plt.xticks(fontproperties=font_prop)
            except Exception:
                # Older matplotlib versions may not accept fontproperties in xticks/yticks
                pass
        else:
            plt.xlabel("Co-occurrence Count", fontsize=12)  # Increase font size
            plt.ylabel("Concept Pairs", fontsize=12)  # Increase font size
            plt.title(f"Top {top_n} Concept Co-occurrences", fontsize=14)  # More dynamic title

        plt.gca().invert_yaxis()  # Display highest count at the top
        plt.tight_layout()

        chart_buffer = io.BytesIO()
        plt.savefig(chart_buffer, format="png")
        plt.close()

        return embed_chart_in_html(chart_buffer, alt_text="Concept Co-occurrence Chart") # 返回base64编码

    except Exception as e:
        logger.error(f"Error generating concept co-occurrence chart: {e}", exc_info=True)
        return ""

def generate_theory_acceptance_chart(theory_acceptance_scores: dict, top_n: int = 10):
    """Generates a horizontal bar chart showing the acceptance scores of the top N theories."""
    top_theories = sorted(theory_acceptance_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    if not top_theories:
        return ""

    try:
        theory_names, acceptance_scores = zip(*top_theories)  # Unpack
        # Filter out None values in theory_names (and corresponding scores)
        valid_data = [(name, score) for name, score in zip(theory_names, acceptance_scores) if name is not None]
        theory_names, acceptance_scores = zip(*valid_data) if valid_data else ([], []) # 确保不为空

        plt.figure(figsize=(10, 6))  # Adjust figure size
        plt.barh(theory_names, acceptance_scores, color="#90ee90")  # Use a softer green

        # Ensure Chinese labels render by using explicit FontProperties when available
        if 'font_prop' in globals() and font_prop is not None:
            plt.xlabel("Acceptance Score (Supporters - Criticisms)", fontsize=12, fontproperties=font_prop)
            plt.ylabel("Theory", fontsize=12, fontproperties=font_prop)
            plt.title(f"Top {top_n} Theories by Acceptance Score", fontsize=14, fontproperties=font_prop)
            try:
                plt.yticks(fontproperties=font_prop)
                plt.xticks(fontproperties=font_prop)
            except Exception:
                pass
        else:
            plt.xlabel("Acceptance Score (Supporters - Criticisms)", fontsize=12)
            plt.ylabel("Theory", fontsize=12)
            plt.title(f"Top {top_n} Theories by Acceptance Score", fontsize=14)

        plt.gca().invert_yaxis()  # Display highest score at the top
        plt.tight_layout()  # Adjust layout

        chart_buffer = io.BytesIO()
        plt.savefig(chart_buffer, format="png")
        plt.close()

        return embed_chart_in_html(chart_buffer, alt_text="Theory Acceptance Chart")# 返回base64编码

    except Exception as e:
        logger.error(f"Error generating theory acceptance chart: {e}", exc_info=True)
        return ""

def generate_collaboration_network_graph(paper_data: list, max_nodes: int = 30):
    """
    Generates an HTML-embedded image of a collaboration network graph using NetworkX.
    Return base64 string
    """
    try:
        graph = nx.Graph()  # Undirected graph for collaborations
        for paper in paper_data:
            kg = paper.get("kg_entities") or {}
            authors = kg.get("researchers", [])
            author_names = [author.get("name") for author in authors if isinstance(author, dict) and author.get("name")]
            # Add edges between all pairs of authors in each paper
            for i in range(len(author_names)):
                for j in range(i + 1, len(author_names)):
                    node1, node2 = author_names[i], author_names[j]
                    if graph.has_edge(node1, node2):
                        graph[node1][node2]["weight"] += 1  # Increment edge weight if it exists
                    else:
                        graph.add_edge(node1, node2, weight=1)  # Add new edge with weight 1

        # Limit the graph size
        if graph.number_of_nodes() > max_nodes:
            # Calculate degree centrality to determine node importance
            degree_centrality = nx.degree_centrality(graph)
            # Select top nodes based on degree centrality
            top_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:max_nodes]
            # Create a subgraph with only the top nodes
            graph = graph.subgraph(top_nodes)
            logger.info(f"Collaboration network graph limited to top {max_nodes} nodes.")

        # Layout the graph (using spring_layout for aesthetics)
        pos = nx.spring_layout(graph, k=0.3, iterations=20, seed=42)

        # Scale edge widths based on weight
        edge_width = [graph[u][v]["weight"] * 0.5 for u, v in graph.edges()]

        # Draw the graph
        plt.figure(figsize=(12, 8))
        ax = plt.gca()  # Get current axes
        # Draw nodes and edges first; draw labels separately so we can pass FontProperties
        nx.draw_networkx_nodes(graph, pos, node_size=800, node_color="#a0cbe2", alpha=0.9, ax=ax)
        nx.draw_networkx_edges(graph, pos, width=edge_width, edge_color="gray", alpha=0.7, ax=ax)
        # Labels: use explicit font properties if available to support Chinese
        if 'font_prop' in globals() and font_prop is not None:
            # Draw labels manually so we can pass FontProperties to matplotlib text
            for node, (x, y) in pos.items():
                ax.text(x, y, str(node), fontproperties=font_prop, fontsize=10, ha='center', va='center')
        else:
            nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax)
        plt.title("Collaboration Network", fontsize=16)

        chart_buffer = io.BytesIO()
        plt.savefig(chart_buffer, format="png", bbox_inches="tight")  # Save without extra whitespace
        plt.close()

        return embed_chart_in_html(chart_buffer, alt_text="Collaboration Network Graph")# 返回base64编码

    except Exception as e:
        logger.error(f"Error generating collaboration network graph: {e}", exc_info=True)
        return ""

def aggregate_analysis(input_file: str, output_file: str, query_terms: list = None):
    """
    综合分析多篇文章的JSON数据，提取综合性属性并生成可视化。
    """
    try:
        # 1. 读取中间文件
        with open(input_file, "r", encoding="utf-8") as f:
            paper_data = [json.loads(line) for line in f]

        # 如果指定了 query_terms，则先过滤数据
        if query_terms:
            logger.info(f"Filtering papers based on query terms: {query_terms}")
            filtered_papers = []
            for paper in paper_data:
                title = (paper.get("title", "") or "").lower()
                abstract = (paper.get("abstract", "") or "").lower()
                authors = " ".join((paper.get("authors", []) or [])).lower()
                search_target = f"{title} {abstract} {authors}"
                if any(term.lower() in search_target for term in query_terms):
                    filtered_papers.append(paper)
            paper_data = filtered_papers  # 使用过滤后的数据
            logger.info(f"After filtering, {len(paper_data)} papers remain.")

        if not paper_data:
            logger.warning("No paper data to aggregate (empty input or filter).")
            analysis_results = {"message": "No data to aggregate."}
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, indent=4, ensure_ascii=False)
            return analysis_results

        # 2. 数据聚合
        researcher_citation_counts = defaultdict(int)
        concept_co_occurrence = Counter()  # 用于统计概念共现次数
        theory_acceptance = defaultdict(lambda: {"supporters": 0, "criticisms": 0})

        for paper in paper_data:
            kg = paper.get("kg_entities") or {}
            # 统计研究者的引用次数 (假设paper 有 citation_count)
            for researcher in kg.get("researchers", []):
                try:
                    name = researcher.get("name") if isinstance(researcher, dict) else str(researcher)
                    if name:
                        researcher_citation_counts[name] += paper.get("citation_count", 0)
                except Exception:
                    # 跳过格式不合预期的条目
                    continue

            # 统计概念的共现次数
            concepts = [c.get("name") for c in (kg.get("concepts", []) or []) if isinstance(c, dict) and c.get("name")]
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    # 使用 frozenset 使概念对的顺序无关紧要
                    concept_pair = tuple(sorted((concepts[i], concepts[j])))
                    concept_co_occurrence[concept_pair] += 1

            # 统计理论的接受程度（支持者和批评者数量）
            for theory in (kg.get("theories", []) or []):
                if isinstance(theory, dict) and theory.get("name"):
                    theory_name = theory.get("name")
                    proponents = theory.get("proponents") if isinstance(theory.get("proponents"), list) else []
                    criticisms = theory.get("criticisms") if isinstance(theory.get("criticisms"), list) else []
                    theory_acceptance[theory_name]["supporters"] += len(proponents)
                    theory_acceptance[theory_name]["criticisms"] += len(criticisms)

        # 3. 计算综合属性
        #   示例：计算研究者的平均引用次数
        total_citations = sum(researcher_citation_counts.values())
        average_citations = total_citations / len(researcher_citation_counts) if researcher_citation_counts else 0

        #   示例：接受程度最高的理论
        theory_acceptance_scores = {
            name: data["supporters"] - data["criticisms"]
            for name, data in theory_acceptance.items()
        }

        # 4. 生成可视化
        concept_cooccurrence_chart_base64 = generate_concept_cooccurrence_chart(concept_co_occurrence)
        theory_acceptance_chart_base64 = generate_theory_acceptance_chart(theory_acceptance_scores)
        collaboration_network_graph_base64 = generate_collaboration_network_graph(paper_data)

        # 5. 整合分析结果
        analysis_results = {
            "average_citations": average_citations,
            "concept_cooccurrence_chart": concept_cooccurrence_chart_base64,
            "theory_acceptance_chart": theory_acceptance_chart_base64,
            "collaboration_network_graph": collaboration_network_graph_base64,
            "message": "分析成功"
        }

        # 6. 保存结果到输出文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=4, ensure_ascii=False)

        logger.info(f"综合分析结果已保存到 {output_file}")
        return analysis_results

    except Exception as e:
        logger.error(f"聚合分析出错: {e}", exc_info=True)
        return {"message": f"分析出错: {e}"}

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="聚合分析JSON数据")
    parser.add_argument("--input_file", type=str, default="all_papers.json", help="输入JSONL文件路径")
    parser.add_argument("--output_file", type=str, default="aggregate_analysis.json", help="输出JSON文件路径")
    parser.add_argument("--query", type=str, help="用于过滤的查询关键词（逗号分隔）")
    return parser.parse_args()

# 主函数
if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.input_file
    output_file = args.output_file
    query_terms = [term.strip() for term in args.query.split(",")] if args.query else None
    analysis_results = aggregate_analysis(input_file, output_file, query_terms)

    if "message" in analysis_results:
        print(f"分析结果: {analysis_results['message']}")