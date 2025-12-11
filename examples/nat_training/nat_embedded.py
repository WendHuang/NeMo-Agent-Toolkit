import asyncio
import sys

from nat.runtime.loader import load_config
from nat.utils import run_workflow


async def amain():
    config = load_config(sys.argv[1])
    query_num = 1
    try:
        while True:
            query = input()
            result = await run_workflow(config=config, prompt=query)
            print(f"Query {query_num}: {query}")
            print(f"Result {query_num}: {result}")
            query_num += 1
    except EOFError:
        pass


asyncio.run(amain())