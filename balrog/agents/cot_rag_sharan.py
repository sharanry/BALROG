import copy
import re
import logging

from balrog.agents.base import BaseAgent
from balrog.client import LLMClientWrapper
from balrog.agents.agent_rag_utils import *
logger = logging.getLogger(__name__)
                           
class CotRagSharanAgent(BaseAgent):
    """An agent that performs actions using a chain-of-thought reasoning process."""

    def __init__(self, client_factory: LLMClientWrapper, prompt_builder, config):
        """Initialize the ChainOfThoughtImprovedAgent with a client, prompt builder, and configuration.

        Args:
            client_factory (LLMClientWrapper): A factory for creating the LLM client instance.
            prompt_builder (PromptBuilder): Object to build prompts for the agent.
            config: Configuration object containing settings for the agent.
        """
        super().__init__(client_factory, prompt_builder)
        self.remember_cot = config.agent.remember_cot
        self.retriever = NethackWikiSearch(config)
        self.retriever.load_index()

    def act(self, obs, prev_action=None):
        """Generate the next action using chain-of-thought reasoning based on the current observation.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the final selected action.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        query_instructions = """
NetHack wiki has the following information:
- game mechanics and optimal strategies 
- characters in the game and their abilities 
- weapons or objects that you find in the game. 

If you want to retrieve information from the wiki, output a concise 4-5 words query sentence of what you would like to retrieve from the wiki. For example: "fountain", or "defeat a fox?" 

<|QUERY|>GENERATED_QUERY<|END|>

Replace GENERATED_QUERY with the generated query. Verify that the query you provided is a valid query from the list of queries given.

In case you want to choose the query "fountain fox", you must output:
<|QUERY|>fountain fox<|END|>

In case you don't want to retrieve any information from the wiki, you must output:
<|NO_QUERY|><|END|>
""".strip()
        
        messages[-1].content += "\n\n" + query_instructions

        query_reasoning = self.client.generate(messages)
        query = self._extract_final_answer(query_reasoning, query=True)

        if query is not None:
            logger.info(f"Extracted question: {query.completion}")
            retrieved_docs = self.retriever.search(query.completion)
        else:
            retrieved_docs = None

        print("Retrieved docs: ", retrieved_docs)

        rag_instructions = f"""
{"Relevant documents:" if retrieved_docs else ""}
{map(lambda x: f"<document>{x[:100]}</document>", enumerate(retrieved_docs)) if retrieved_docs else ""}

First, think about the best course of action.
Then, you must choose exactly one of the listed actions and output it strictly in the following format:

<|ACTION|>YOUR_CHOSEN_ACTION<|END|>

Replace YOUR_CHOSEN_ACTION with the chosen action. Verify that the action you provided is a valid action from the list of actions given.

In case you want to choose the action "go forward", you must output:
<|ACTION|>go forward<|END|>""".strip()


        # Add the updated instructions to the last message
        messages[-1].content += "\n\n" + rag_instructions

        # Generate the CoT reasoning
        cot_reasoning = self.client.generate(messages)

        # Extract the final answer from the CoT reasoning
        final_answer = self._extract_final_answer(cot_reasoning)

        return final_answer

    def _extract_final_answer(self, reasoning, query=False):
        """Extract the final action from the chain-of-thought reasoning response.

        Args:
            reasoning (LLMResponse): The response containing CoT reasoning and action.

        Returns:
            LLMResponse: The response with the extracted final action in `completion`
                         and the entire chain-of-thought in `reasoning`.
        """
        # Make a copy so we don't mutate the original
        final_answer = copy.deepcopy(reasoning)

        # Store the entire chain-of-thought (raw completion) in `reasoning`
        final_answer = final_answer._replace(reasoning=reasoning.completion)

        # Now parse the strict action format: <|ACTION|> ... <|END|>
        completion_text = reasoning.completion
        if query:
            match = re.search(r"<\|QUERY\|>(.*?)<\|END\|>", completion_text, re.DOTALL)
        else:
            match = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion_text, re.DOTALL)
        
        if match:
            extracted_action = match.group(1).strip()
        else:
            # Fallback to the entire completion if not matched
            if query:
                extracted_action = None
            else:
                extracted_action = "Failed to obtain a valid action from the reasoning."

        # Replace the final `completion` with only the extracted action
        if extracted_action is None:
            return None
        final_answer = final_answer._replace(completion=extracted_action)

        return final_answer