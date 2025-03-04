# BALROG Experimental Setup

This directory contains a simple experimental setup using BALROG agents to compare different LLM behaviors.

## Setup

1. Make sure you have BALROG installed and set up in your environment.

2. Set up the required API keys as environment variables:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export ANTHROPIC_API_KEY="your_anthropic_api_key"
   # export HF_API_KEY="your_huggingface_api_key"  # If using HuggingFace models
   ```

3. The experiment is configured in `balrog_experiment.py` and includes:
   - A simple math task to demonstrate agent behavior
   - Support for multiple agent types (OpenAI, Anthropic, HuggingFace)
   - Result collection and comparison

## Running the Experiment

To run the experiment:

```bash
python balrog_experiment.py
```

The script will:
1. Initialize the configured agents
2. Run a simple math task
3. Collect and display responses from each agent
4. Include metadata about the responses

## Modifying the Experiment

You can modify the experiment by:
1. Changing the task in `create_simple_task()`
2. Adding or removing agents in the `main()` function
3. Modifying the evaluation criteria
4. Adding more complex conversation flows

## Notes

- Make sure your API keys are properly set up before running the experiment
- The example uses a simple math problem, but you can modify it for more complex tasks
- You can uncomment additional agents in the code to test with different models 