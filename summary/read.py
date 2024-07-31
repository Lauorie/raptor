import os
import time
import json
import tiktoken
from tqdm import tqdm
from loguru import logger
from vllm_llm import LLM
from prompts import PROMPTS
from typing import List, Tuple, Optional

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
            logger.info(f"Splitting the text into {len(text_chunks)} chunks to be processed.")
            logger.info(f"Chunk lengths are {[len(self.tokenize(x)) for x in text_chunks]}")

        # set system message
        system_message_content = PROMPTS.READING_SYSTEM_PROMPT.format(no_response="无答案。")
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
            user_message_content = PROMPTS.READING_USER_PROMPT.format(ref_doc=user_message_content, instruction=query)
            
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
                accumulated_results.append(response)
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error occurred: {e}")
                continue
            accumulated_results.append(response)        
        print('\n\n'.join(accumulated_results))
        
        content = self._extract_content(query, accumulated_results)
        logger.info(f"content: {content}")
        final_messages = [
            {"role": "system", "content": PROMPTS.KNOWLEDGE_BASE_SYSTEM_PROMPT},
            {"role": "user", "content": PROMPTS.KNOWLEDGE_BASE_USER_PROMPT.format(context=content,query=query)}
        ]
        return self.llm.streaming_answer(final_messages)
    
    
    def _extract_content(self, query: str, accumulated_results: List[str]) -> str:
        res = []
        invalid_answers = {'无答案。', 'none', 'None', 'no', 'No'}

        for result in accumulated_results:
            try:
                parsed_result = json.loads(result)
                if not isinstance(parsed_result, dict):
                    continue

                content = parsed_result.get('content', '').strip()
                result_value = parsed_result.get('res', '').lower()

                if content and result_value not in invalid_answers:
                    res.append(f"问题：{query} 的答案：{content}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from result: {result}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error processing result: {result}", exc_info=True)

        return " ".join(res)

if __name__ == '__main__':
    file_path = '/root/web_demo/HybirdSearch/es_app_0702/data.txt'
    reader = SlowReader(file_path)
    query = '本文是否研究了 contrastive learning，如果是请把文章名称和摘要列出来，如果不是请回答：本文没有研究 contrastive learning。'
    answer = reader.read(query=query,detail=1,minimum_chunk_size=8192)
    res = "".join(answer)
    logger.info(f'关于你的问题：‘{query}’，我的答案是：\n{res}')
    
