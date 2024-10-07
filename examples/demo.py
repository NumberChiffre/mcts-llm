import math
import os

import dspy
from dotenv import load_dotenv

from mcts_llm.mctsr import MCTSr, MultipleTurnSelfRefine, Policy, ZeroShotCoT

load_dotenv()


if __name__ == "__main__":
    ollama = dspy.OllamaLocal(
        model="qwen2.5:7b-instruct",
        model_type="chat",
        temperature=1.0,
        max_tokens=1024,
        num_ctx=1024,
        timeout_s=600,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    dspy.settings.configure(lm=ollama, experimental=True)

    problem = "Alice has 3 sisters and she also has 4 brothers. How many sisters does Alice’s brother have?"
    answer = "4"
    print(f"problem: {problem}")
    print(f"Answer: {answer}")

    zero_shot_cot = ZeroShotCoT()
    zero_shot_answer = zero_shot_cot(problem).answer
    print(f"Zero-Shot answer: {zero_shot_answer}")

    multiple_turn_self_refine = MultipleTurnSelfRefine(num_turns=1)
    multiple_turn_self_refine_answer = multiple_turn_self_refine(problem).answer
    print(f"Multiple-Turn Self-Refine answer: {multiple_turn_self_refine_answer}")

    mctsr = MCTSr(c=math.sqrt(2), samples_per_node=4, policy=Policy.GREEDY)
    mctsr_answer = mctsr(problem).answer
    print(f"MCStr answer: {mctsr_answer}")
