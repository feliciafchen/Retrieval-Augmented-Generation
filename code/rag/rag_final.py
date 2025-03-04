import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from openai import OpenAI

from tqdm import tqdm

#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

#### CONFIG PARAMETERS END---

class ChunkExtractor:

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Improvement: Retrieve all results concurrently with one call to ray.get()
        results = ray.get(ray_response_refs)

        # Aggregate results by interaction_id
        from collections import defaultdict
        chunk_dictionary = defaultdict(list)
        for interaction_id, _chunks in results:
            chunk_dictionary[interaction_id].extend(_chunks)

        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)
        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids

import faiss  # make sure to install faiss-cpu or faiss-gpu as appropriate

class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge.
    """
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None):
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()

        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def calculate_embeddings(self, sentences):
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        return embeddings

    def get_batch_size(self) -> int:
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Extract chunks using ChunkExtractor.
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        # Calculate embeddings for chunks and queries.
        chunk_embeddings = self.calculate_embeddings(chunks).astype("float32")
        query_embeddings = self.calculate_embeddings(queries).astype("float32")

        batch_retrieval_results = []
        for idx, interaction_id in enumerate(batch_interaction_ids):
            query_embedding = query_embeddings[idx].reshape(1, -1)
            # Identify chunks corresponding to the current interaction id.
            relevant_idx = (chunk_interaction_ids == interaction_id).nonzero()[0]
            if len(relevant_idx) == 0:
                batch_retrieval_results.append([])
                continue

            # Extract relevant embeddings and chunks.
            relevant_embeddings = chunk_embeddings[relevant_idx]
            relevant_chunks = chunks[relevant_idx]

            # Build a FAISS index (using inner product, since embeddings are normalized).
            d = relevant_embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(relevant_embeddings)
            # Search for the top-N most similar chunks.
            topk = NUM_CONTEXT_SENTENCES
            distances, indices = index.search(query_embedding, topk)
            retrieved_chunks = relevant_chunks[indices[0]]
            batch_retrieval_results.append(retrieved_chunks)

        formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results)

        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=formatted_prompts[0],
                n=1,
                top_p=0.9,
                temperature=0.1,
                max_tokens=50,
            )
            answers = [response.choices[0].message.content]
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,
                    top_p=0.9,
                    temperature=0.1,
                    skip_special_tokens=True,
                    max_tokens=50,
                ),
                use_tqdm=False
            )
            answers = [response.outputs[0].text for response in responses]

        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        system_prompt = (
            "You are provided with a question and various references. Your task is to answer the question "
            "succinctly, using the fewest words possible. If the references do not contain the necessary "
            "information to answer the question, respond with 'I don't know'. There is no need to explain "
            "the reasoning behind your answers."
        )
        formatted_prompts = []
        for idx, query in enumerate(queries):
            query_time = query_times[idx]
            retrieval_results = batch_retrieval_results[idx]

            references = ""
            if len(retrieval_results) > 0:
                references += "# References \n"
                for snippet in retrieval_results:
                    references += f"- {snippet.strip()}\n"
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]

            user_message = f"{references}\n------\n\n"
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts
