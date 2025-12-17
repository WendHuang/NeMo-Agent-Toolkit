import logging
import os

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LlamaIndexRAGConfig(FunctionBaseConfig, name="retail_sales_rag"):

    llm_name: LLMRef = Field(description="The name of the LLM to use for the RAG engine.")
    embedder_name: EmbedderRef = Field(description="The name of the embedder to use for the RAG engine.")
    data_dir: str = Field(description="The directory containing the data to use for the RAG engine.")
    description: str = Field(description="A description of the knowledge included in the RAG system.")
    collection_name: str = Field(default="context", description="The name of the collection to use for the RAG engine.")


def _walk_directory(root: str):
    for root, dirs, files in os.walk(root):
        for file_name in files:
            yield os.path.join(root, file_name)


@register_function(config_type=LlamaIndexRAGConfig, framework_wrappers=[LLMFrameworkEnum.LLAMA_INDEX])
async def retail_sales_rag_tool(config: LlamaIndexRAGConfig, builder: Builder):
    from llama_index.core import Settings
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import StorageContext
    from llama_index.core import VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
    embedder = await builder.get_embedder(config.embedder_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)

    Settings.embed_model = embedder
    Settings.llm = llm

    files = list(_walk_directory(config.data_dir))
    docs = SimpleDirectoryReader(input_files=files).load_data()
    logger.info("Loaded %s documents from %s", len(docs), config.data_dir)

    parser = SentenceSplitter(
        chunk_size=400,
        chunk_overlap=20,
        separator=" ",
    )
    nodes = parser.get_nodes_from_documents(docs)

    index = VectorStoreIndex(nodes)

    query_engine = index.as_query_engine(similarity_top_k=3, )

    async def _arun(inputs: str) -> str:
        """
        Search product catalog for information about tablets, laptops, and smartphones
        Args:
            inputs: user query about product specifications
        """
        try:
            response = query_engine.query(inputs)
            return str(response.response)

        except Exception as e:
            logger.error("RAG query failed: %s", e)
            return f"Sorry, I couldn't retrieve information about that product. Error: {str(e)}"

    yield FunctionInfo.from_fn(_arun, description=config.description)
