from textwrap import dedent
from .tokens import BEGIN, END

def dynamic_mode_preamble() -> str:
    return f'''You should use functions to help handle the real time user queries. Remember:
    Before retrieving any functions, carefully analyze the user's query
    and break the task into smaller steps or subgoals. Use this
    decomposition to decide which functions you will need. After this
    planning step, describe the required functions using the special
    tokens format. The following sections outline three key rules for doing this, including how to use the special tokens format:
    1. **Dynamic Function Retrieval:**
    - The only function available at the start is the "Finish" function.
    - To call any other function, you **must** first retrieve it from a function corpus based on your needs.
    - Even if the user query suggests a possible function, you **must** still perform retrieval to obtain the actual function you can call.

    2. **Retrieval Process:**
    - Do not call a function(tool) to perform the retrieval
    - To retrieve the required function, simply write the following in your output:
        {BEGIN}  [Describe the function you need in detail here]  {END}
    - If the user query suggests a possible function, you may **include that hint in your description**, but you must still perform retrieval.
    - If you need to retrieve **multiple functions**, describe each function **separately** using its own `{BEGIN} ... {END}` block.
    - Once you write this description, **pause and wait** until the system retrieves the available functions for you.
    - Do **not** assume or infer the function's existence without performing retrieval first.

    3. **Task Completion and Error Handling:**
    - You **must** always call the "Finish" function at the end of the task.
    - The final output must be self-contained and complete, and should include all information a user would reasonably need, without assuming any follow-up questions.
    - If you cannot handle the task or if function calls consistently fail (e.g., the retrieved function is invalid), use Finish->give_up_and_restart.
    - **Do not use "Finish" at the beginning** — it is only for when the task is fully completed or must be restarted.'''

def static_mode_preamble(tool_lines: list[str]) -> str:
    header = 'You should use fuctions to help handle the real time user querys. Remember:\n' \
             '1.ALWAYS call "Finish" function at the end of the task. And the final answer should contain enough information to show to the user, If you can\'t handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.\n' \
             '2.Do not use origin tool names, use only subfunctions\' names.\n' \
             'You have access of the following tools:\n'
    return header + "\n".join(tool_lines)

# Description by description refinement
def make_dbd_refine_messages(
    *, query: str, ancestor_desc: str, current_desc: str, lineage_examples_block: str, turn: int
):
    """Refine a single pseudo-function description within a specific lineage."""
    content = dedent(f"""
        You are assisting an autonomous agent that has already decomposed the overall task into multiple subtasks.
        The agent must retrieve functions (tools) from a function database and then call them.
        Your role is to refine exactly one function description for the current subtask only. Do not propose or add additional functions; other subtasks will be optimized separately.

        Context:
        - That agent received the following Global user query(for reference only): 
        {query}

        - Draft function description to refine (this is the ONLY function description you should output, refined):
        {current_desc}

        - Ad-hoc retrieved function descriptions (examples from the function database, from prior retrievals in THIS subtask).
        Write each example on a new line; keep in mind these are database-style descriptions:
        {lineage_examples_block}

        Goal:
        Your goal is to refine the provided function description so that it 
        (1) closely matches the style, tone, and structure of the retrieved function examples, and 
        (2) is optimized for retrieval — meaning you may adapt or rephrase the description, based on your understanding of the current subtask and the actual example functions existing in the database, to maximize the chance of retrieving the most suitable function for solving this subtask.

        Output RULES (STRICT):
        - Output **exactly one** block using the special tokens:
          {BEGIN}
          ...your single refined function description here...
          {END}
        - No commentary before/after the block. No extra blocks.
        - The refined description should closely follow the style, tone, and formatting of the retrieved examples in function database.
    """).strip()

    return [{"role": "user", "content": content}]


def make_dbd_regen_messages(
    *, query: str, ancestor_desc: str, lineage_examples_block: str
):
    """Regenerate a single pseudo-tool description for a specific lineage."""
    content = dedent(f"""
        You are regenerating a single pseudo-tool description for a specific pseudo-tool lineage.

        Context:
        - Task query (binding target): 
        {query}

        - Ancestor pseudo-tool (generation 0 of this lineage):
        {ancestor_desc}

        - Ad-hoc retrieved tool descriptions (examples from the tool database, from prior retrievals in THIS lineage).
        Write each example on a new line; keep in mind these are database-style descriptions:
        {lineage_examples_block}

        Goal:
        Regenerate the pseudo-tool description for this lineage so it best matches the examples’ style/semantics and serves the task query.

        Output RULES (STRICT):
        - Output **exactly one** block using the special tokens:
          {BEGIN}
          ...your single regenerated pseudo-tool description here...
          {END}
        - No commentary before/after the block. No extra blocks.
        - The description should focus on capabilities/parameters the retriever could match, not natural language plans.
    """).strip()

    return [{"role": "user", "content": content}]

def build_scattershot_seeded_messages(*, query: str, ancestor_desc: str, exemplar_block: str):
    """
    Generate ONE diversified pseudo-tool description for the SAME lineage intent,
    guided by the ancestor and exemplar blurbs. Caller loops 'size' times.
    Output must be exactly one <|begin_func_description|> ... <|end_func_description|> block.
    """
    content = dedent(f"""
    You are assisting an autonomous agent that has already decomposed the overall task into multiple subtasks.
    The agent must retrieve functions (tools) from a function database and then call them.
    Your role is to generate exactly one diversified function description for the current subtask only. Do not propose or add additional functions; other subtasks will be optimized separately.

    Context:
    - That agent received the following Global user query(for reference only): 
    {query}

    - Draft function description to diversify (This function description is only for refering to the subtask, you should NOT copy it):
    {ancestor_desc}

    - Ad-hoc retrieved function descriptions (examples from the function database, from prior retrievals in THIS subtask).
    Write each example on a new line; keep in mind these are database-style descriptions:
    {exemplar_block}

    Goal:
    Your goal is to generate a new function description that:
    (1) Preserves the intent of the current subtask.  
    (2) Closely follows the style, tone, and structure of the retrieved function examples.  
    (3) Is optimized for retrieval — you may adapt or rephrase the description, based on your understanding of the current subtask and the actual example functions in the database, to maximize the chance of retrieving the most suitable function.  
    (4) Introduces variation in phrasing, emphasis, or perspective compared to previous generations, so that repeated calls produce diversified descriptions.

    Output RULES (STRICT):
    - Output **exactly one** block using the special tokens:
        {BEGIN}
        ...your single refined function description here...
        {END}
    - No commentary before/after the block. No extra blocks.
    - The refined description should closely follow the style, tone, and formatting of the retrieved examples in function database.
    - Vary wording each time (we will call you multiple times) for diversification.
    """).strip()
    return [{"role": "user", "content": content}]

FORMAT_INSTRUCTIONS_SYSTEM=f"""
    You are AutoGPT, you can use many tools(functions) to do the following task. In this workflow you must perform a **single, comprehensive analysis** of the user query.
    You only have one chance to analyze the query, so you must consider everything at once.
    If you determine that one or more functions are needed, you must output **all of them at the same time**, and you must use the special retrieval format for **each function** separately.
    The following sections outline two key rules for doing this, including how to use the special tokens format:
    1. **Dynamic Function Retrieval:**
    - You does not have access to any functions at the start.
    - To call any function, you **must** first retrieve it from a function corpus based on your needs.
                     
    2. **Retrieval Process:**
    - To retrieve the required function, simply write the following in your output:
        {BEGIN}  [Describe the function you need in detail here]  {END}
    - If the user query suggests a possible function, you may **include that hint in your description**, but you must still perform retrieval.
    - If you need to retrieve **multiple functions**, describe each function **separately** using its own `{BEGIN} ... {END}` block.
    - Once you finish writing all these descriptions, **pause and wait** until the system retrieves the available functions for you.
    - Do **not** assume or infer the function's existence without performing retrieval first.
    """

FORMAT_INSTRUCTIONS_USER = """
Task: {input_description}
Begin!
"""
