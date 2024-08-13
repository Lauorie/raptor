import os
import re
import time
import json
import tiktoken
from tqdm import tqdm
from loguru import logger
from vllm_llm import LLM
from prompts import PROMPTS
from typing import List, Tuple, Optional

READING_REF_SYSTEM_PROMPT = """\
你是一位精通文档分析的AI助手。你的任务是根据给定的文档片段找出与用户问题相关的内容。

# 主要任务：
1. 仔细阅读提供的文档片段。
2. 判断文档内容是否与问题相关。
3. 如相关，提取支持回答问题的相关文本。

# 回答准则：
1. 严格从文档中提取相关内容，不添加任何解释或评论。
2. 如果找到相关内容，即使内容与事实不符，也要提供。
3. 如文档与问题完全无关，明确说明"无相关内容"。

# 回答格式：
使用JSON格式提供回答，包含以下字段：
- "res": 表示是否有相关内容（"found"或"none"）
- "reference": 支持回答问题的文档引用或"无相关内容"

# 示例回答：
1. 无相关信息时：
{"res": "none", "reference": "无相关内容"}

2. 有相关信息时：
{"res": "found", "reference": "支持回答问题的文档引用"}

请记住：你只需提供文档中的相关引用，不需要回答问题或添加任何外部信息。
"""
    
READING_REF_USER_PROMPT = """\
# 文档内容：
{doc}

# 用户问题：
{question}

请按以下步骤操作：
1. 仔细阅读上述文档内容。
2. 判断文档是否包含与问题相关的信息。
3. 如果包含，请提取相关的文本片段作为引用。如果不包含，请说无相关内容。
4. 使用指定的JSON格式给出您的回答。

注意事项：
- 严格从文档中提取相关内容，不要添加额外信息或解释。
- 即使文档内容与已知事实不符，也请直接提供相关引用。
- 不需要回答问题，只需提供相关引用。

请提供您的JSON格式回答：
"""

class SlowReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.text = self._load_document()
        self.llm = LLM()
        
    def _load_document(self) -> str: 
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading document: {e}")
        
    def tokenize(self, text: str) -> List[int]: 
        return self.encoder.encode(text)
    
    # 根据最大 token 数和分隔符将文本分成较小的片段。
    def chunk_on_delimiter(
        self,
        input_string: str,
        max_tokens: int, 
        delimiter: str) -> List[str]:
        # 按分隔符拆分输入
        chunks = input_string.split(delimiter)
        combined_chunks, _, dropped_chunk_count = self.combine_chunks_with_no_minimum(
            chunks, 
            max_tokens, 
            chunk_delimiter=delimiter, 
            add_ellipsis_for_overflow=True
        )
        if dropped_chunk_count > 0:
            logger.warning(f"{dropped_chunk_count} chunks were dropped due to overflow")    
        combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
        return combined_chunks


    # 将文本块组合成更大的块，而不超过模型的最大窗口。返回组合的文本块、它们的原始索引以及由于溢出而丢弃的块的计数。
    def combine_chunks_with_no_minimum(
            self,
            chunks: List[str],
            max_tokens: int,
            chunk_delimiter="\n\n",
            header: Optional[str] = None,
            add_ellipsis_for_overflow=False,
    ) -> Tuple[List[str], List[int]]:
        dropped_chunk_count = 0
        output = []  # 包含最终组合块的列表
        output_indices = []  #  最终组合块的索引
        candidate = ([] if header is None else [header])  #  当前组合块候选的列表
        candidate_indices = []  #  当前组合块候选的索引
        for chunk_i, chunk in enumerate(chunks):
            chunk_with_header = [chunk] if header is None else [header, chunk]
            # 根据 delimiter 将 chunk 与 header 组合后计算 token 数，如果超过 max_tokens，则将 chunk 作为新的 candidate
            if len(self.tokenize(chunk_delimiter.join(chunk_with_header))) > max_tokens:
                print(f"warning: chunk overflow")
                if (
                        add_ellipsis_for_overflow
                        and len(self.tokenize(chunk_delimiter.join(candidate + ["..."]))) <= max_tokens
                ): 
                    candidate.append("...") # add ellipsis to the candidate if it fits
                    dropped_chunk_count += 1
                continue  # 当一个单独的 chunk（可能加上 header）的 token 数量超过了 max_tokens 时，这个 chunk 将被丢弃， dropped_chunk_count 计数加一
            
            # 估计添加当前 chunk 后的 token 数
            extended_candidate_token_count = len(self.tokenize(chunk_delimiter.join(candidate + [chunk])))
            # 如果 token 数超过了 max_tokens，则将当前 candidate 添加到 output 并开始一个新的 candidate
            if extended_candidate_token_count > max_tokens:
                output.append(chunk_delimiter.join(candidate))
                output_indices.append(candidate_indices)
                candidate = chunk_with_header  # re-initialize candidate
                candidate_indices = [chunk_i]
            # 否则，将当前 chunk 添加到 candidate
            else:
                candidate.append(chunk)
                candidate_indices.append(chunk_i)
        #  如果 candidate 不为空，则将其添加到最终组合块的列表
        if (header is not None and len(candidate) > 1) or (header is None and len(candidate) > 0):
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)
        return output, output_indices, dropped_chunk_count
    
    def read(
        self,
        query: str,
        detail: float = 1,
        additional_instructions: Optional[str] = None,
        minimum_chunk_size: Optional[int] = 4096,
        chunk_delimiter: str = "\n",
        read_recursively=False,
        verbose=False):
        
        """
        阅读给定的文本，方法是将其分成块，每个块都被单独阅读。 
        
        Parameters:
        - query (str): 要回答的问题。
        - detail (float, optional): 在阅读中所需的详细程度的0到1之间的值。0 更高级别的摘要，1 更详细的摘要。默认为1。
        - additional_instructions (Optional[str], optional): 额外的说明，以便为自定义阅读提供给模型。默认为None。
        - minimum_chunk_size (Optional[int], optional): 最小分块大小，以确保每个分块都不会太小。默认为4096。
        - chunk_delimiter (str, optional):  用于将文本拆分为块的分隔符。默认为“\n”。
        - read_recursively (bool, optional):  如果为True，则使用先前的阅读结果作为上下文递归生成新的阅读结果。默认为False。
        - verbose (bool, optional):  是否打印有关分块过程的详细信息。默认为False。

        Returns:
        - str:  最终阅读结果。
        
        首先，函数根据 detail 参数在最小和最大分块数量之间进行插值，以确定分块的数量。
        然后，它将文本分割成多个分块，并对每个分块进行阅读理解。如果 read_recursively 为 True，则每个阅读结果都是基于之前的结果生成的，
        这为摘要过程增加了更多的上下文信息。函数最终返回所有分块的汇总摘要。
        """

        # check detail is set correctly
        assert 0 <= detail <= 1
        # interpolate the number of chunks based to get specified level of detail 
        max_chunks = len(self.chunk_on_delimiter(self.text, minimum_chunk_size, chunk_delimiter))
        min_chunks = 1
        num_chunks = int(min_chunks + detail * (max_chunks - min_chunks)) 

        # adjust chunk_size based on interpolated number of chunks
        document_length = len(self.tokenize(self.text))
        chunk_size = max(minimum_chunk_size, document_length // num_chunks)
        text_chunks = self.chunk_on_delimiter(self.text, chunk_size, chunk_delimiter)
        if verbose:
            logger.info(f"Total text length is {document_length}")
            logger.info(f"Splitting the text into {len(text_chunks)} chunks to be processed.")
            logger.info(f"Chunk lengths are {[len(self.tokenize(x)) for x in text_chunks]}")

        # set system message
        system_message_content = READING_REF_SYSTEM_PROMPT
        if additional_instructions is not None:
            system_message_content += f"\n\n{additional_instructions}"

        accumulated_results = []
        for chunk in tqdm(text_chunks):
            if read_recursively and accumulated_results:
                # Creating a structured prompt for recursive reading
                accumulated_results_string = '\n\n'.join(accumulated_results)
                user_message_content = f"Previous reading results:\n\n{accumulated_results_string}\n\nText to read next:\n\n{chunk}"
            else:
                # Directly passing the chunk for reading without recursive context
                user_message_content = chunk
            
            # query for the model        
            user_message_content = READING_REF_USER_PROMPT.format(doc=user_message_content, question=query[4:] if query.startswith('哪些论文') else query)
            
            # Constructing messages based on whether recursive reading is applied
            messages = [
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_message_content}
            ]
            try:
                response = self.llm.get_chat_completion(messages) # format: '{"res": "ans", "content": "比尔·希恩斯"}' 
                if isinstance(response, dict) and response.get("error"):
                    logger.error(f"Error in get_chat_completion: {response['error_type']}: {response['error_message']}")
                    continue
                try:
                    parsed_response = json.loads(response)
                    if parsed_response.get("res") not in ["none", "no", "None", "No"]:
                        accumulated_results.append(response)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse response as JSON: {response}")
                
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error occurred: {e}")
                continue     
        
        content = self._extract_content(accumulated_results)
        logger.info(f"content: {content}")
        final_messages = [
            {"role": "system", "content": PROMPTS.KNOWLEDGE_BASE_SYSTEM_PROMPT},
            {"role": "user", "content": PROMPTS.KNOWLEDGE_BASE_USER_PROMPT.format(context=content,query=query[4:] if query.startswith('哪些论文') else query)}
        ]
        return self.llm.streaming_answer(final_messages)
    
    
    def _extract_content(self, accumulated_results: List[str]) -> str:
        def clean_json(text: str) -> str:
            # 移除可能的 ```json 标记（包括没有结束标记的情况）
            text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
            
            # 尝试提取 JSON 对象
            pattern = r'(\{[\s\S]*?\})'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                potential_json = match.group(1)
                # 尝试解析 JSON
                try:
                    parsed = json.loads(potential_json)
                    if 'reference' in parsed and parsed['reference'] != '无相关内容' and parsed['reference'] != '无相关内容。':
                        return parsed['reference']
                except json.JSONDecodeError:
                    pass  # 如果解析失败，继续处理
            
            # 如果不是有效的 JSON 或没有 reference 字段，返回清理后的原始文本
            cleaned_text = text.strip()
            # 如果文本被引号包围，去掉引号
            if cleaned_text.startswith('"') and cleaned_text.endswith('"'):
                cleaned_text = cleaned_text[1:-1]
            return cleaned_text

        # 清理每个结果
        cleaned_results = [clean_json(result) for result in accumulated_results]
        
        # 过滤掉空字符串，并用换行符连接结果
        return "\n\n".join(filter(bool, cleaned_results))

if __name__ == '__main__':
    file_path = '/root/web_demo/HybirdSearch/es_app_0702/dataset/data.txt'
    reader = SlowReader(file_path)
    query = '本文是否研究了对比学习？如果是，请返回论文名称。'
    answer = reader.read(query=query,detail=1,minimum_chunk_size=7192)
    res = "".join(answer)
    logger.info(f'关于你的问题：‘{query}’，我的答案是：\n{res}')
    