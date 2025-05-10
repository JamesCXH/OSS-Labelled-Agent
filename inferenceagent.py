import anthropic
import re
import os
import string
import json
import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
# from groq import Groq
# from together import Together
# from openai import OpenAI
# from cerebras.cloud.sdk import Cerebras
# import google.generativeai as google_client
from pydantic import BaseModel

###############################################################################
# GLOBAL CONSTANTS
###############################################################################

# Maximum number of times we'll retry JSON decoding with the same model before fallback
MAX_JSON_DECODE_RETRIES = 2

# For provider calls. If the call to LLM fails at a network/SDK level, we fallback to Claude Sonnet
FALLBACK_MODEL = "claude-3-5-sonnet-latest"

###############################################################################
# DATACLASSES
###############################################################################

@dataclass
class LLMMessage:
    """
    Data class representing a single message input for an LLM call.
    message_role: "system" or "user"
    content: The text content for that role.
    """
    message_role: str
    content: str


@dataclass
class AgentCall:
    """
    Data class representing the result of an LLM call.
    - messages: The list of messages that formed the prompt for this call.
    - llm_response: Raw response from the LLM.
    - parsed_output: Any structured result we parse from llm_response (optional).
    """
    messages: List[LLMMessage]           # UPDATED: store the entire list of messages
    llm_response: Any
    parsed_output: Optional[Any] = None


###############################################################################
# INITIALIZE CLIENTS
###############################################################################

# Initialize API clients outside of functions to avoid re-initialization
anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# groq_client = Groq()
#
# together_client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

# openai_client = OpenAI()

# cerebras_client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
#
# google_client.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

###############################################################################
# LLM CALL FUNCTION WITH ERROR/FALLBACK LOGIC
###############################################################################

async def call_llm(
    messages: List[LLMMessage],
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-latest",
    max_tokens: int = 15000,
) -> AgentCall:
    """
    Asynchronously calls the specified LLM provider with the given messages (a list of LLMMessage),
    with fallback logic:
      (1) If the LLM call fails at the API level, fallback to Anthropic Sonnet.
      (2) Return an AgentCall containing the raw response plus the original messages.
    """

    # --- 1) Attempt primary call, if fail => fallback
    try:
        output = await _provider_llm_call(messages, provider, model, max_tokens)
    except Exception as e:
        print(f"[call_llm] Error with provider={provider}, model={model} => Fallback to Anthropic Sonnet.\nError: {e}")
        # Fallback call with Anthropic's claude sonnet
        fallback_provider = "anthropic"
        fallback_model = FALLBACK_MODEL
        output = await _provider_llm_call(messages, fallback_provider, fallback_model, max_tokens)

    # Return an AgentCall for consistency
    return AgentCall(
        messages=messages,        # UPDATED
        llm_response=output,
        parsed_output=None,
    )


async def _provider_llm_call(
    messages: List[LLMMessage],
    provider: str,
    model: str,
    max_tokens: int,
) -> str:
    """
    Internal helper that performs the actual call to each provider's API.
    Called by call_llm. Raises an exception if something goes wrong.
    """

    def anthropic_call() -> str:
        system_segments = []
        user_segments = []

        for msg in messages:
            if msg.message_role == "system":
                system_segments.append({"type": "text", "text": msg.content})
            elif msg.message_role == "user":
                user_segments.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg.content}
                    ]
                })

        # Combine system text
        system_messages_list = []
        if system_segments:
            system_messages_list = [
                {
                    "type": "text",
                    "text": "\n".join([seg["text"] for seg in system_segments])
                }
            ]

        # Combine user text
        user_messages_list = []
        if user_segments:
            combined_user_text = ""
            for seg in user_segments:
                for c in seg["content"]:
                    combined_user_text += c["text"] + "\n"
            user_messages_list = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": combined_user_text.strip()}]
                }
            ]

        message = anthropic_client.messages.create(
            model=model,
            max_tokens=8192, # PRESET FOR SONNET AS OF DEC 2024
            temperature=0,
            system=system_messages_list,
            messages=user_messages_list
        )
        return message.content[0].text

    # def groq_call() -> str:
    #     final_messages = []
    #     for msg in messages:
    #         final_messages.append({
    #             "role": msg.message_role,
    #             "content": msg.content
    #         })
    #
    #     completion = groq_client.chat.completions.create(
    #         model=model,
    #         messages=final_messages,
    #         temperature=0,
    #         max_tokens=max_tokens,
    #         top_p=1,
    #         stream=False,
    #         stop=None,
    #     )
    #     return completion.choices[0].message.content
    #
    # def together_call() -> str:
    #     final_messages = []
    #     for msg in messages:
    #         if msg.message_role == "user":
    #             final_messages.append({
    #                 "role": "user",
    #                 "content": msg.content
    #             })
    #
    #     response = together_client.chat.completions.create(
    #         model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    #         messages=final_messages,
    #         max_tokens=max_tokens,
    #         temperature=0,
    #         top_p=1,
    #         top_k=50,
    #         repetition_penalty=1,
    #         stop=["<|eot_id|>"],
    #         stream=False,
    #     )
    #     return response.choices[0].message.content
    #
    # def openai_call(openai_model: str = "gpt-4o", store: bool = False) -> str:
    #     openai_msgs = []
    #     for msg in messages:
    #         openai_msgs.append({
    #             "role": msg.message_role,
    #             "content": msg.content
    #         })
    #
    #     completion = openai_client.chat.completions.create(
    #         model=openai_model,
    #         temperature=0,
    #         max_tokens=max_tokens,
    #         messages=openai_msgs,
    #         store=store,
    #         metadata={"agent": "action", "testing": "testing"} if store else None
    #     )
    #     return completion.choices[0].message.content
    #
    # def cerebras_call() -> str:
    #     c_msgs = []
    #     for msg in messages:
    #         c_msgs.append({
    #             "role": msg.message_role,
    #             "content": msg.content
    #         })
    #
    #     completion = cerebras_client.chat.completions.create(
    #         messages=c_msgs,
    #         model=model,
    #         stream=False,
    #         max_tokens=max_tokens,
    #         temperature=0,
    #         top_p=1
    #     )
    #     return completion.choices[0].message.content
    #
    # def google_call() -> str:
    #     system_text = ""
    #     user_text = ""
    #     for msg in messages:
    #         if msg.message_role == "system":
    #             system_text += msg.content + "\n"
    #         else:
    #             user_text += msg.content + "\n"
    #
    #     generation_config = {
    #         "temperature": 0,
    #         "top_p": 1,
    #         "top_k": 40,
    #         "max_output_tokens": 8192,
    #         "response_mime_type": "text/plain",
    #     }
    #
    #     model_instance = google_client.GenerativeModel(
    #         model_name="gemini-exp-1114",
    #         generation_config=generation_config,
    #         system_instruction=system_text.strip()
    #     )
    #
    #     chat_session = model_instance.start_chat(history=[])
    #     return chat_session.send_message(user_text.strip()).text

    if provider == "anthropic":
        return await asyncio.to_thread(anthropic_call)
    # elif provider == "groq":
    #     return await asyncio.to_thread(groq_call)
    # elif provider == "together":
    #     return await asyncio.to_thread(together_call)
    # elif provider == "openai":
    #     return await asyncio.to_thread(lambda: openai_call(openai_model="gpt-4o", store=False))
    # elif provider == 'openai-o1-preview-store':
    #     return await asyncio.to_thread(lambda: openai_call(openai_model="o1-preview", store=True))
    # elif provider == 'openai-o1-mini-store':
    #     return await asyncio.to_thread(lambda: openai_call(openai_model="o1-mini", store=True))
    # elif provider == 'openai-store':
    #     return await asyncio.to_thread(lambda: openai_call(openai_model="gpt-4o", store=True))
    # elif provider == "cerebras":
    #     return await asyncio.to_thread(cerebras_call)
    # elif provider == "google":
    #     return await asyncio.to_thread(google_call)
    else:
        raise ValueError("Invalid provider. Choose among 'anthropic', 'groq', 'together', 'openai', 'cerebras', 'google'.")


###############################################################################
# HELPER FUNCTION: TRY DECODING JSON, ELSE RETRY OR FALLBACK
###############################################################################

async def try_json_parse(
    llm_response: str,
    parse_func,
    messages: List[LLMMessage],
    provider: str,
    model: str,
    parse_error_message: str,
    max_retries: int = MAX_JSON_DECODE_RETRIES,
) -> Any:
    """
    Attempt to parse JSON from llm_response with parse_func.
    If it fails up to `max_retries` times, we fallback to Anthropic Sonnet using the same messages,
    then parse again. If still fails, raise an error.
    """
    # 1) Try parse once
    try:
        return parse_func(llm_response)
    except Exception as e:
        print(f"{parse_error_message} => JSON parsing error: {e}")

    # 2) If parse fails, attempt up to `max_retries` more calls with the *same provider & model*
    for i in range(max_retries):
        print(f"Retrying JSON parse with the same model/provider (Attempt {i+1}/{max_retries})...")
        agent_call = await call_llm(
            messages=messages,
            provider=provider,
            model=model
        )
        llm_response = agent_call.llm_response
        try:
            return parse_func(llm_response)
        except Exception as e:
            print(f"{parse_error_message} => JSON parsing error (attempt {i+1}): {e}")

    # 3) If still failing, fallback to anthropic sonnet
    print(f"{parse_error_message} => Falling back to Anthropic Sonnet model.")
    fallback_messages = messages
    fallback_provider = "anthropic"
    fallback_model = FALLBACK_MODEL

    fallback_call = await call_llm(
        messages=fallback_messages,
        provider=fallback_provider,
        model=fallback_model
    )
    llm_response = fallback_call.llm_response
    try:
        return parse_func(llm_response)
    except Exception as e:
        # If STILL fails, raise an error
        raise ValueError(f"{parse_error_message} => Even fallback failed to provide valid JSON. Last error: {e}")

###############################################################################
# STRUCTURED CALL EXAMPLE (call_4o_structured_action)
###############################################################################

async def call_4o_structured_action(
    system_prompt: str = '',
    user_prompt: str = '',
    max_tokens: int = 10000
) -> AgentCall:
    """
    Example of a structured call to OpenAI's Beta parse method.
    We'll keep a single system & user message for demonstration.
    """

    # Build our messages
    messages = [
        LLMMessage(message_role="system", content=system_prompt),
        LLMMessage(message_role="user", content=user_prompt),
    ]

    def call_structured_4o():
        class ReasoningStep(BaseModel):
            step_number: int
            thought: str
            conclusion: str

        class ChosenAction(BaseModel):
            action_number: int
            action_reason: str

        class ChainOfThought(BaseModel):
            steps: List[ReasoningStep]
            chosen_action_sequence: List[ChosenAction]

        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            temperature=0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ChainOfThought
        )
        return completion

    completion = await asyncio.to_thread(call_structured_4o)

    chosen_actions = [
        (
            chosen.action_number,
            chosen.action_reason
        ) for chosen in completion.choices[0].message.parsed.chosen_action_sequence
    ]

    return AgentCall(
        messages=messages,   # UPDATED
        llm_response=completion.choices[0].message,
        parsed_output=chosen_actions
    )


###############################################################################
# BELOW: EXAMPLES OF AGENT CALLS, UPDATED TO USE THE NEW call_llm MESSAGES
###############################################################################

async def call_action_part1(
        task: str,
        task_notes: str,
        scrape_tree_no_special: str,
        action_memory: List,
        context: str,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet-latest"
):
    memory_without_qa = '***\n'
    for i, lin_mem in enumerate(action_memory):
        if not lin_mem.is_question:
            memory_without_qa += f"{i + 1})\nINTENT: {lin_mem.intent}\nEFFECT: {lin_mem.location_details}\nREASONING: {lin_mem.difference_reasoning}\n***\n"

    def read_system_prompt():
        with open('prompts/chained_action_decider/chain_part1_system.txt', 'r') as f:
            return f.read()

    system_prompt_template = await asyncio.to_thread(read_system_prompt)
    system_prompt = system_prompt_template

    def read_user_prompt():
        with open('prompts/chained_action_decider/chain_part1_user.txt', 'r') as f:
            return f.read()

    user_prompt_template = await asyncio.to_thread(read_user_prompt)
    user_replacements = {
        'ax_tree': scrape_tree_no_special,
        'task': task,
        'task_notes': task_notes,
        'memory': memory_without_qa,
    }
    user_prompt = string.Template(user_prompt_template).substitute(user_replacements)

    # Build LLM messages
    messages = [
        LLMMessage("system", system_prompt),
        LLMMessage("user", user_prompt),
    ]

    agent_call = await call_llm(
        messages=messages,
        provider=provider,
        model=model
    )
    output = agent_call.llm_response

    print("PART 1 CALL BEGIN")
    print(system_prompt)
    print(user_prompt)
    print(output)
    print("PART 1 CALL END")

    def parse_grounded_progress_summary(response_text: str) -> str:
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        if json_matches:
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    if 'grounded_progress_summary' in data:
                        return data.get('grounded_progress_summary', '')
                except json.JSONDecodeError as e:
                    raise ValueError(f"Action Chain 1: JSON decoding failed - {e}")
        raise ValueError("Action Chain 1: No valid JSON block found or missing 'grounded_progress_summary' key.")

    parsed_output = await try_json_parse(
        llm_response=agent_call.llm_response,
        parse_func=parse_grounded_progress_summary,
        messages=messages,
        provider=provider,
        model=model,
        parse_error_message="Action Chain 1"
    )

    agent_call.parsed_output = parsed_output
    return agent_call


async def call_action_part2(
        task: str,
        task_notes: str,
        summarized_info: str,
        curr_inf_tree: str,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet-latest"
):
    def read_user_prompt():
        with open('prompts/chained_action_decider/chain_part2_user.txt', 'r') as f:
            return f.read()

    user_prompt_template = await asyncio.to_thread(read_user_prompt)
    user_replacements = {
        'ax_tree': curr_inf_tree,
        'task': task,
        'task_notes': task_notes,
        'first_chain_output': summarized_info,
    }
    user_prompt = string.Template(user_prompt_template).substitute(user_replacements)

    def read_system_prompt():
        with open('prompts/chained_action_decider/chain_part2_system.txt', 'r') as f:
            return f.read()

    system_prompt_template = await asyncio.to_thread(read_system_prompt)
    system_prompt = system_prompt_template

    messages = [
        LLMMessage("system", system_prompt),
        LLMMessage("user", user_prompt)
    ]

    agent_call = await call_llm(
        messages=messages,
        provider=provider,
        model=model
    )
    output = agent_call.llm_response
    print("PART 2 CALL BEGIN")
    print(system_prompt)
    print(user_prompt)
    print(output)
    print("PART 2 CALL END")

    def parse_action_number_reason(response_text: str) -> Tuple[str, str]:
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        if json_matches:
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    if 'action_number' in data and 'action_reason' in data:
                        return (data['action_number'], data['action_reason'])
                except json.JSONDecodeError as e:
                    raise ValueError(f"Action Chain 2: JSON decoding failed - {e}")
        raise ValueError("Action Chain 2: No valid JSON block found or missing keys 'action_number', 'action_reason'.")

    parsed_output = await try_json_parse(
        llm_response=agent_call.llm_response,
        parse_func=parse_action_number_reason,
        messages=messages,
        provider=provider,
        model=model,
        parse_error_message="Action Chain 2"
    )

    agent_call.parsed_output = parsed_output
    return agent_call


async def call_action_agent(
    task: str,
    ax_tree: str,
    action_memory: List,
    context: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-latest"
) -> AgentCall:
    new_action_memory = '\n***'
    for i, lin_mem in enumerate(action_memory):
        if not lin_mem.is_question:
            new_action_memory += f"{i + 1})\nINTENT: {lin_mem.intent}\nEFFECT: {lin_mem.location_details}\nREASONING: {lin_mem.difference_reasoning}\n***\n"

    def read_system_prompt():
        with open('prompts/action_decider/action_decider_llama_system.txt', 'r') as f:
            return f.read()

    system_prompt_template = await asyncio.to_thread(read_system_prompt)
    replacements = {
        'context': context
    }
    system_prompt = string.Template(system_prompt_template).substitute(replacements)

    def read_user_prompt():
        with open('prompts/action_decider/action_decider_llama_user.txt', 'r') as f:
            return f.read()

    user_prompt_template = await asyncio.to_thread(read_user_prompt)
    replacements = {
        'ax_tree': ax_tree,
        'task': task,
        'action_memory': new_action_memory,
    }
    user_prompt = string.Template(user_prompt_template).substitute(replacements)

    messages = [
        LLMMessage("system", system_prompt),
        LLMMessage("user", user_prompt),
    ]
    print("\nACTION CALL BEGIN\n")
    print(user_prompt)
    print(system_prompt)
    print("\nACTION CALL END\n")

    agent_call = await call_llm(
        messages=messages,
        provider=provider,
        model=model
    )
    print("LLM Response:\n", agent_call.llm_response)

    pattern = r'choose\(\s*(\d+),\s*"(.*)"\)'
    match = re.search(pattern, agent_call.llm_response)

    parsed_output: Optional[Tuple[int, str]] = None
    if match:
        parsed_output = (int(match.group(1)), match.group(2))

    agent_call.parsed_output = parsed_output
    return agent_call


async def call_action_agent_multi(
    task: str,
    ax_tree: str,
    action_memory: List,
    context: str,
    provider: str = 'openai'
) -> AgentCall:
    agent_call = None
    new_action_memory = ''
    for i, lin_mem in enumerate(action_memory):
        new_action_memory += f"\n{i + 1})\n{i + 1}) EFFECT: {lin_mem.location_details}"

    print('*' * 80)
    print(ax_tree)

    if provider == 'openai':
        def read_user_prompt():
            with open('prompts/action_decider/action_decider_multi_user_OAI.txt', 'r') as f:
                return f.read()

        user_prompt_template = await asyncio.to_thread(read_user_prompt)
        replacements = {
            'ax_tree': ax_tree,
            'task': task,
            'action_memory': new_action_memory,
        }
        user_prompt = string.Template(user_prompt_template).substitute(replacements)

        def read_system_prompt():
            with open('prompts/action_decider/action_decider_multi_system_OAI.txt', 'r') as f:
                return f.read()

        system_prompt_template = await asyncio.to_thread(read_system_prompt)
        replacements = {
            'context': context
        }
        system_prompt = string.Template(system_prompt_template).substitute(replacements)

        messages = [
            LLMMessage("system", system_prompt),
            LLMMessage("user", user_prompt),
        ]
        # We'll keep calling call_4o_structured_action for demonstration
        agent_call = await call_4o_structured_action(system_prompt, user_prompt)

    elif provider == 'anthropic':
        def read_user_prompt():
            with open('prompts/action_decider/action_decider_multi_user_anthropic.txt', 'r') as f:
                return f.read()

        user_prompt_template = await asyncio.to_thread(read_user_prompt)
        replacements = {
            'ax_tree': ax_tree,
            'task': task,
            'action_memory': new_action_memory,
        }
        user_prompt = string.Template(user_prompt_template).substitute(replacements)

        def read_system_prompt():
            with open('prompts/action_decider/action_decider_multi_system_anthropic.txt', 'r') as f:
                return f.read()

        system_prompt_template = await asyncio.to_thread(read_system_prompt)
        replacements = {
            'context': context
        }
        system_prompt = string.Template(system_prompt_template).substitute(replacements)

        messages = [
            LLMMessage("system", system_prompt),
            LLMMessage("user", user_prompt),
        ]
        agent_call = await call_llm(messages=messages, provider=provider)

        output = agent_call.llm_response
        print(output)
        json_pattern = re.compile(r'```json\s*(\{.*?}|\[.*?])\s*```', re.DOTALL | re.MULTILINE)
        match = json_pattern.search(output)
        action_tuples = []

        if not match:
            raise ValueError("No JSON block found in the LLM output.")

        json_str = match.group(1)
        try:
            actions = json.loads(json_str)
            for action in actions:
                if 'action_number' in action and 'action_reason' in action:
                    tuple_entry = (action['action_number'], action['action_reason'])
                    action_tuples.append(tuple_entry)
        except json.JSONDecodeError as jde:
            print(f"JSON Decode Error: {jde}")
        except KeyError as ke:
            print(f"Key Error: {ke}")

        agent_call.parsed_output = action_tuples

    return agent_call



async def call_memory_agent(
    web_agent_task: str,
    action_memory: List,
    world_memory: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-latest"
) -> AgentCall:
    new_action_memory = ''
    for i, lin_mem in enumerate(action_memory):
        new_action_memory += f"\n{i + 1})\n{i + 1}) EFFECT: {lin_mem.location_details}"

    def read_prompt():
        with open('prompts/world_mem_prompt_v2.txt', 'r') as f:
            return f.read()

    prompt_template = await asyncio.to_thread(read_prompt)
    replacements = {
        'web_agent_task': web_agent_task,
        'world_memory': world_memory,
        'action_memory': new_action_memory,
    }
    prompt = string.Template(prompt_template).substitute(replacements)

    messages = [LLMMessage("user", prompt)]
    agent_call = await call_llm(
        messages=messages,
        provider=provider,
        model=model
    )

    print("LLM Response:\n", agent_call.llm_response)

    json_match = re.search(r'\{[\s\S]*}', agent_call.llm_response)
    parsed_output = "Error: No JSON object found in the answer"

    if json_match:
        json_str = json_match.group(0)
        try:
            parsed_json = json.loads(json_str)
            new_world_memory = parsed_json.get("new_world_memory")
            if new_world_memory is not None:
                parsed_output = new_world_memory
            else:
                print("Error: 'new_world_memory' key not found in JSON")
                parsed_output = "Error: 'new_world_memory' key not found in JSON"
        except json.JSONDecodeError:
            print("Error: Invalid JSON format")
            parsed_output = "Error: Invalid JSON format"

    agent_call.parsed_output = parsed_output
    return agent_call


async def call_reflect_agent(
    reason_for_action: str,
    old_ax_tree: str,
    new_ax_tree: str,
    web_agent_task: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-latest"
) -> AgentCall:
    def read_user_prompt():
        with open('prompts/reflect/reflect_llama_user.txt', 'r') as f:
            return f.read()

    def read_system_prompt():
        with open('prompts/reflect/reflect_llama_system.txt', 'r') as f:
            return f.read()

    user_prompt_template = await asyncio.to_thread(read_user_prompt)
    replacements = {
        'reason_for_action': reason_for_action,
        'old_accessibility_tree': old_ax_tree,
        'new_accessibility_tree': new_ax_tree,
        'web_agent_task': web_agent_task,
    }
    user_prompt = string.Template(user_prompt_template).substitute(replacements)

    system_prompt_str = await asyncio.to_thread(read_system_prompt)

    messages = [
        LLMMessage("system", system_prompt_str),
        LLMMessage("user", user_prompt),
    ]

    agent_call = await call_llm(
        messages=messages,
        provider=provider,
        model=model
    )

    print("LLM Response:\n", agent_call.llm_response)

    def parse_reflect_json(response_text: str) -> Tuple[str, str]:
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        if json_matches:
            for json_str in json_matches:
                try:
                    cleaned_json = json_str.replace("\\'", "'")
                    data = json.loads(cleaned_json)
                    if 'final_answer' in data:
                        action_effect = data.get('final_answer', '')
                        difference_reasoning = data.get('difference_reasoning', '')
                        return (action_effect, difference_reasoning)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Reflect Restore Error: JSON decoding failed - {e}")
        raise ValueError("Reflect Restore Error: No valid JSON block or missing 'final_answer' key in the response.")

    parsed_output = await try_json_parse(
        llm_response=agent_call.llm_response,
        parse_func=parse_reflect_json,
        messages=messages,
        provider= "anthropic",
        model="claude-3-5-sonnet-latest",
        parse_error_message="Reflect Restore"
    )

    agent_call.parsed_output = parsed_output
    return agent_call


async def call_task_separator(
    web_agent_task: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-latest"
) -> AgentCall:
    def read_prompt():
        with open('prompts/task_separator_prompt_json.txt', 'r') as f:
            return f.read()

    prompt_template = await asyncio.to_thread(read_prompt)
    replacements = {
        'web_agent_task': web_agent_task,
    }
    prompt = string.Template(prompt_template).substitute(replacements)

    messages = [LLMMessage("user", prompt)]
    agent_call = await call_llm(
        messages=messages,
        provider=provider,
        model=model
    )

    answer = agent_call.llm_response.strip()
    print("LLM Response:\n", answer)

    def parse_task_separator_json(response_text: str) -> Optional[List[Any]]:
        json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        match = re.search(json_pattern, response_text)
        if match:
            json_str = match.group(0)
            parsed_json = json.loads(json_str)
            return parsed_json.get('items', [])
        raise ValueError("Task Separator: No valid JSON object found or 'items' key missing.")

    parsed_output = await try_json_parse(
        llm_response=answer,
        parse_func=parse_task_separator_json,
        messages=messages,
        provider=provider,
        model=model,
        parse_error_message="Task Separator parse"
    )

    agent_call.parsed_output = parsed_output
    return agent_call


async def call_input_agent(
    user_task: str,
    agent_intent: str,
    input_ax_tree: str,
    context: str,
    task_notes: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-latest"
) -> AgentCall:
    def read_prompt():
        with open('prompts/mass_input_prompt_json.txt', 'r') as f:
            return f.read()

    prompt_template = await asyncio.to_thread(read_prompt)
    replacements = {
        'user_task': user_task,
        'agent_intent': agent_intent,
        'input_ax_tree': input_ax_tree,
        'task_notes': task_notes
    }
    prompt = string.Template(prompt_template).substitute(replacements)

    messages = [LLMMessage("user", prompt)]
    agent_call = await call_llm(
        messages=messages,
        provider=provider,
        model=model
    )

    print("INPUT CALL BEGIN")
    print(prompt)
    print("INPUT CALL END")

    def parse_input_json(response_text: str) -> List[Tuple[int, str]]:
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        if not json_matches:
            raise ValueError("No JSON found in LLM response")

        data = json.loads(json_matches[0])
        return [
            (choice['text_area_number'], choice['desired_input'])
            for choice in data.get('choices', [])
        ]

    parsed_output = await try_json_parse(
        llm_response=agent_call.llm_response,
        parse_func=parse_input_json,
        messages=messages,
        provider=provider,
        model=model,
        parse_error_message="Mass input parse"
    )

    agent_call.parsed_output = parsed_output
    return agent_call


async def call_unified_task_clarifier(
    user_task: str,
    context: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-latest"
) -> AgentCall:
    def read_prompt():
        with open('prompts/unified_task_clarifier_json.txt', 'r') as f:
            return f.read()

    prompt_template = await asyncio.to_thread(read_prompt)
    replacements = {
        'user_task': user_task,
        'context': context
    }
    prompt = string.Template(prompt_template).substitute(replacements)

    messages = [LLMMessage("user", prompt)]
    agent_call = await call_llm(
        messages=messages,
        provider=provider,
        model=model
    )

    print("LLM Response:\n", agent_call.llm_response)

    def parse_unified_task_clarifier(response_text: str) -> List[str]:
        json_matches = re.findall(r'\[.*?\]', response_text, re.DOTALL)
        for json_str in json_matches:
            parsed_json = json.loads(json_str)
            if isinstance(parsed_json, list) and all(isinstance(q, str) for q in parsed_json):
                return parsed_json
        raise ValueError("Unified Task Clarifier Error: No valid JSON array found in the output.")

    parsed_output = await try_json_parse(
        llm_response=agent_call.llm_response,
        parse_func=parse_unified_task_clarifier,
        messages=messages,
        provider=provider,
        model=model,
        parse_error_message="Unified Task Clarifier parse"
    )

    agent_call.parsed_output = parsed_output
    return agent_call


async def call_unified_question_cleaner(
    user_task: str,
    user_qa: List[Tuple[str, str]],
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-latest"
) -> AgentCall:
    def read_prompt():
        with open('prompts/unified_question_cleaner_json.txt', 'r') as f:
            return f.read()

    prompt_template = await asyncio.to_thread(read_prompt)

    formatted_user_qa = ''
    for question, answer in user_qa:
        formatted_user_qa += f'Question: {question.strip()}\nAnswer: {answer.strip()}\n'

    replacements = {
        'formatted_user_qa': formatted_user_qa,
        'user_task': user_task
    }
    prompt = string.Template(prompt_template).substitute(replacements)
    print("User Prompt:\n", prompt)

    messages = [LLMMessage("user", prompt)]
    agent_call = await call_llm(
        messages=messages,
        provider=provider,
        model=model
    )
    print("LLM Response:\n", agent_call.llm_response)

    def parse_unified_question_cleaner(response_text: str) -> str:
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, response_text)
        if json_matches:
            for json_str in json_matches:
                data = json.loads(json_str)
                if 'new_task' in data:
                    return data.get('new_task', '')
        raise ValueError("Question Cleaner Error: No valid JSON or missing 'new_task' key in the LLM response.")

    parsed_output = await try_json_parse(
        llm_response=agent_call.llm_response,
        parse_func=parse_unified_question_cleaner,
        messages=messages,
        provider=provider,
        model=model,
        parse_error_message="Unified Question Cleaner parse"
    )

    agent_call.parsed_output = parsed_output
    return agent_call


async def call_action_pruner(
    user_task: str,
    ax_tree: str,
    action_memory: List,
    actions,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-latest"
) -> AgentCall:
    def read_user_prompt():
        with open('prompts/prune/prune_user.txt', 'r') as f:
            return f.read()

    def read_system_prompt():
        with open('prompts/prune/prune_system.txt', 'r') as f:
            return f.read()

    user_prompt_template = await asyncio.to_thread(read_user_prompt)
    system_prompt_template = await asyncio.to_thread(read_system_prompt)

    replacements = {
        'task': user_task,
        'ax_tree': ax_tree,
        'action_memory': action_memory,
        'actions': actions
    }
    user_prompt = string.Template(user_prompt_template).substitute(replacements)
    system_prompt = system_prompt_template

    print("User Prompt:\n", user_prompt)
    print("System Prompt:\n", system_prompt)

    messages = [
        LLMMessage("system", system_prompt),
        LLMMessage("user", user_prompt),
    ]
    agent_call = await call_llm(
        messages=messages,
        provider=provider,
        model=model
    )

    print("LLM Response:\n", agent_call.llm_response)

    match = re.search(r'\[\s*(-?\d+\s*(,\s*-?\d+\s*)*)?\]', agent_call.llm_response)
    if match:
        list_content = match.group(0)
        try:
            parsed = eval(list_content)
            agent_call.parsed_output = parsed
            return agent_call
        except (SyntaxError, ValueError):
            pass

    agent_call.parsed_output = []
    return agent_call


async def call_intermediate_questions_agent(
    user_task: str,
    ax_tree: str,
    question_intent: str,
    task_notes: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-latest"
) -> AgentCall:
    def read_user_prompt():
        with open('prompts/question_agent/user.txt', 'r') as f:
            return f.read()

    def read_system_prompt():
        with open('prompts/question_agent/system.txt', 'r') as f:
            return f.read()

    user_prompt_template = await asyncio.to_thread(read_user_prompt)
    system_prompt_template = await asyncio.to_thread(read_system_prompt)

    replacements = {
        'ax_tree': ax_tree,
        'task': user_task,
        'question_intent': question_intent,
        'task_notes': task_notes
    }
    user_prompt = string.Template(user_prompt_template).substitute(replacements)
    system_prompt = system_prompt_template

    messages = [
        LLMMessage("system", system_prompt),
        LLMMessage("user", user_prompt),
    ]

    agent_call = await call_llm(
        messages=messages,
        provider=provider,
        model=model
    )

    print("LLM Response:\n", agent_call.llm_response)

    def parse_intermediate_questions(response_text: str) -> List[str]:
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        if json_matches:
            for json_str in json_matches:
                data = json.loads(json_str)
                if 'questions' in data:
                    return data.get('questions', [])
        raise ValueError("intermediate questions agent error: No valid JSON or missing 'questions' key.")

    parsed_output = await try_json_parse(
        llm_response=agent_call.llm_response,
        parse_func=parse_intermediate_questions,
        messages=messages,
        provider=provider,
        model=model,
        parse_error_message="Intermediate Questions"
    )

    agent_call.parsed_output = parsed_output
    return agent_call


async def call_unified_notes_cleaner(
    user_task: str,
    task_notes: str,
    user_qa: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-latest"
) -> AgentCall:
    def read_user_prompt():
        with open('prompts/task_notes_cleaner.txt', 'r') as f:
            return f.read()

    user_prompt_template = await asyncio.to_thread(read_user_prompt)
    replacements = {
        'user_task': user_task,
        'original_task_notes': task_notes,
        'new_user_answers': user_qa,
    }
    user_prompt = string.Template(user_prompt_template).substitute(replacements)
    print(user_prompt)

    messages = [LLMMessage("user", user_prompt)]
    agent_call = await call_llm(
        messages=messages,
        provider=provider,
        model=model
    )

    print("LLM Response:\n", agent_call.llm_response)

    def parse_unified_notes_cleaner(response_text: str) -> str:
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        if json_matches:
            for json_str in json_matches:
                data = json.loads(json_str)
                if 'new_task_notes' in data:
                    return data.get('new_task_notes', "")
        raise ValueError("task notes cleaner: No valid JSON or missing 'new_task_notes' key.")

    parsed_output = await try_json_parse(
        llm_response=agent_call.llm_response,
        parse_func=parse_unified_notes_cleaner,
        messages=messages,
        provider=provider,
        model=model,
        parse_error_message="Task Notes Cleaner"
    )

    agent_call.parsed_output = parsed_output
    return agent_call


async def call_unified_context_cleaner(
    user_task: str,
    task_notes: str,
    context: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-latest"
) -> AgentCall:

    def read_user_prompt():
        with open('prompts/context_cleaner.txt', 'r') as f:
            return f.read()

    user_prompt_template = await asyncio.to_thread(read_user_prompt)
    replacements = {
        'user_task': user_task,
        'original_task_notes': task_notes,
        'context': context,
    }
    user_prompt = string.Template(user_prompt_template).substitute(replacements)
    # Call the updated call_llm asynchronously
    print(user_prompt)
    messages = [LLMMessage("user", user_prompt)]
    agent_call = await call_llm(
        messages=messages,
        provider=provider,
        model=model
    )

    print("LLM Response:\n", agent_call.llm_response)

    def parse_unified_context_cleaner(response_text: str) -> str:
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'

        json_matches = re.findall(json_pattern, agent_call.llm_response, re.DOTALL)
        parsed_output: Optional[str] = ""

        if json_matches:
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    if 'new_task_notes' in data:
                        return data.get('new_task_notes', [])
                except json.JSONDecodeError as e:
                    print(f"context agent cleaner: JSON decoding failed for a matched block - {e}")
                    continue
        else:
            print("context agent error: No JSON object found in the LLM response")
    # Update the parsed_output in AgentCall
    parsed_output = await try_json_parse(
        llm_response=agent_call.llm_response,
        parse_func=parse_unified_context_cleaner,
        messages=messages,
        provider=provider,
        model=model,
        parse_error_message="Task Notes Cleaner"
    )

    agent_call.parsed_output = parsed_output

    return agent_call