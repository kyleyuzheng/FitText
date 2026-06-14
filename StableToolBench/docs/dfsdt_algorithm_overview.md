# DFSDT: Depth-First Search Decision Tree for Tool-Augmented LLM Agents

## Appendix: Algorithm and System Design

In this appendix, we describe the inference pipeline used to evaluate tool-augmented LLM agents on open-domain question answering tasks. The system connects a large language model (LLM) to a corpus of real-world APIs through a structured tree search process, enabling multi-step reasoning, dynamic tool retrieval, and iterative refinement of tool selection.

---

## 1. System Overview

The system implements an end-to-end pipeline that, given a natural language query, identifies relevant API tools, conducts a depth-first tree search over reasoning trajectories, and produces a final answer grounded in API observations. The architecture comprises five principal components:

1. **Query Processing Layer** -- Loads evaluation tasks, filters previously completed queries, and prepares the execution environment.
2. **Tool Retrieval Module** -- Identifies candidate API tools from a large corpus, either statically (once, before search) or dynamically (during search, triggered by LLM reasoning).
3. **DFSDT Search Engine** -- The core algorithm that explores a tree of reasoning steps (thoughts, actions, observations) via depth-first traversal with backtracking.
4. **LLM Interface** -- Manages communication with the language model using an OpenAI-compatible function-calling protocol, including message construction, token budgeting, and retry logic.
5. **Meta-Strategy Layer** -- An optional outer loop that orchestrates multiple DFSDT passes with evolutionary or sampling-based strategies to improve tool retrieval quality.

The data flow proceeds as follows:

```
Query --> [Tool Retrieval] --> [DFSDT Tree Search] --> [Answer Extraction]
               ^                      |
               |                      v
               +--- [Dynamic Re-Retrieval] <--- [LLM Pseudo-Tool Generation]
```

---

## 2. Query Processing Pipeline

The pipeline processes queries through the following stages:

### 2.1 Task Loading and Filtering

The system ingests a set of evaluation queries, each consisting of a natural language question and, optionally, a ground-truth list of relevant APIs. Before execution, the pipeline checks for existing output files and skips previously completed queries to support resumable evaluation runs. The task list is shuffled with a fixed random seed to distribute workload evenly across parallel workers.

### 2.2 Tool Corpus Preparation

Two retrieval modes govern how tools are made available to the agent:

- **Static retrieval**: Before the tree search begins, the retrieval module identifies a fixed set of candidate APIs based on the query. These tools are converted into the LLM's function-calling schema and remain unchanged throughout the search.
- **Dynamic retrieval**: The search begins with no pre-loaded tools. Instead, the LLM is instructed to reason about what tools it needs by generating structured pseudo-tool descriptions. These descriptions trigger retrieval during the search, and retrieved tools are added to the function schema incrementally.

In both modes, a terminal action schema ("Finish") is appended to the function list, enabling the agent to signal task completion.

### 2.3 Environment Instantiation

For each query, the system instantiates an API execution environment that wraps the tool corpus, manages the function schema, and handles API invocations. This environment is responsible for:

- Maintaining a registry mapping normalized function names to their underlying API endpoints.
- Executing API calls against either a local tool server or a remote API gateway.
- Classifying API response status codes (e.g., success, timeout, rate limit, authorization failure).
- Truncating overly long observations to respect token limits.

### 2.4 Answer Generation and Serialization

The tree search produces a terminal node containing the agent's final answer. The pipeline serializes the complete search tree, the winning trajectory, and metadata (token counts, validity flags) into a structured output file. If no valid answer is found, the system falls back to a randomly selected "give up" node, if any exist.

---

## 3. DFSDT Algorithm

The Depth-First Search Decision Tree (DFSDT) is the core reasoning algorithm. It explores a tree of interleaved LLM reasoning steps and API observations, using depth-first traversal with controlled backtracking.

### 3.1 Tree Structure

Each node in the search tree represents one atomic unit of the agent's reasoning chain. Nodes are typed as follows:

| Node Type        | Description |
|------------------|-------------|
| **Thought**      | A free-text reasoning step produced by the LLM, often containing analysis of the current state or pseudo-tool descriptions for dynamic retrieval. |
| **Action**       | The name of a function (API tool) the LLM has chosen to invoke. |
| **Action Input** | The arguments passed to the selected function, along with the resulting observation from the API execution environment. |

A complete reasoning step typically consists of a Thought node (optional), followed by one or more Action/Action Input pairs generated from a single LLM call via parallel function calling.

The root node is initialized with:
- A **system message** containing the task preamble (which differs between static and dynamic retrieval modes).
- A **user message** containing the natural language query.

### 3.2 Node Expansion

At each node, the algorithm solicits a response from the LLM by presenting the full message history accumulated along the path from root to the current node. The LLM response is parsed into one of two forms:

1. **Text with pseudo-tool descriptions** (dynamic mode): If the response contains structured description blocks (delimited by special tokens), the system interprets these as requests for tool retrieval. A Thought node is created, retrieval is triggered, and the newly retrieved tools are added to the function schema for subsequent LLM calls.

2. **Function calls**: If the response contains one or more tool invocations, the system creates an Action node and an Action Input node for each call. The API execution environment processes each call and returns an observation, which is appended to the message history.

When a node already has children from prior expansion attempts, a **diversity prompt** is injected. This prompt presents summaries of previously explored children (including their actions, arguments, observations, and estimated quality scores) and instructs the LLM to generate a substantively different alternative.

### 3.3 Planner Gate

At the root node, the system may optionally invoke a dedicated **planning model** -- a separate, typically more capable reasoning LLM -- to produce the first reasoning step. This planner generates an initial plan or set of pseudo-tool descriptions before the primary LLM takes over for subsequent expansion steps. The planner is invoked at most once per search.

### 3.4 Backtracking

The algorithm employs a controlled backtracking mechanism governed by two parameters:

- **Prune back length**: When a node is pruned (due to an invalid action, API error, or the agent choosing to "give up and restart"), the algorithm backtracks a fixed number of levels (default: 2) up the tree before attempting alternative expansions.
- **Final answer back length**: When a terminal answer is found, the algorithm backtracks by a similar amount, allowing exploration of alternative branches at higher levels if multiple answers are desired.

A node is pruned under the following conditions:
- The maximum search depth has been reached.
- The API returns a hallucinated (non-existent) function name (status code 1).
- The agent explicitly invokes the Finish action with a "give up and restart" directive (status code 4).

When a hallucinated function name is detected, the function name in the message history is replaced with a sentinel value, preventing the LLM from repeating the same hallucination in sibling branches.

### 3.5 Termination

The search terminates when any of the following conditions are met:

1. **Answer found**: The LLM invokes the Finish action with a `give_answer` return type and provides a final answer string. The corresponding node is marked as terminal.
2. **Answer quota reached**: The algorithm has found the requested number of valid answers (default: 1).
3. **Query budget exhausted**: The total number of LLM calls exceeds a hard limit, triggering an immediate exit.
4. **Depth limit reached**: A branch reaches the maximum allowed depth without finding an answer.

### 3.6 Filtered vs. Unfiltered DFS

The algorithm supports two traversal variants:

- **DFSDT (unfiltered, `with_filter=False`)**: Each child node is immediately expanded via recursive DFS. This is the standard mode and produces a single deep trajectory before backtracking.
- **Filtered DFS (`with_filter=True`)**: All children at a given level are generated first, then ranked by an auxiliary LLM evaluation. Children are expanded in order of decreasing estimated quality. This variant incurs additional LLM calls for ranking but may find better solutions faster.

### 3.7 Parameters

| Parameter              | Role                                              | Typical Value |
|------------------------|---------------------------------------------------|---------------|
| `single_chain_max_step`| Maximum depth of any branch in the tree           | 15            |
| `tree_beam_size`       | Number of children generated per node per layer    | 1--3          |
| `max_query_count`      | Hard limit on total LLM invocations               | 20            |
| `answer`               | Number of terminal answers to find before stopping | 1             |

### 3.8 Pseudocode

```
Algorithm: DFSDT(node, max_depth, beam_width, max_queries, answers_needed)

    if node.depth >= max_depth or node.is_pruned or node.is_terminal:
        if node.is_terminal:
            record answer; return FINAL_ANSWER_BACK_LENGTH
        else:
            mark node as pruned; return PRUNE_BACK_LENGTH

    for i = 1 to beam_width:
        if node has existing children:
            inject diversity prompt summarizing prior children

        if at root and planner is available and unused:
            response <-- planner_LLM(node.messages)
            mark planner as used
        else:
            response <-- primary_LLM(node.messages, available_tools)

        if response contains pseudo-tool descriptions:
            create Thought child node
            if dynamic retrieval mode:
                check tool memory for duplicates
                if not duplicate:
                    trigger retrieval; add tools to function schema
                    update tool memory

        if response contains function calls:
            for each function call in response:
                create Action node (function name)
                create Action Input node (arguments)
                observation <-- execute API call
                record observation and status code
                if status == HALLUCINATION:
                    replace function name with sentinel
                if status == TERMINAL_ANSWER:
                    mark node as terminal

        backtrack_distance <-- DFSDT(child, max_depth, beam_width, max_queries, answers_needed)
        if enough answers found:
            return LARGE_VALUE (exit all recursion)
        if backtrack_distance > 1:
            return backtrack_distance - 1

    return 1  (backtrack one level)
```

---

## 4. Dynamic Tool Retrieval

In dynamic retrieval mode, the agent does not begin with a fixed tool set. Instead, tool discovery is integrated into the search process itself.

### 4.1 Retrieval Trigger

During node expansion, when the LLM produces a response containing one or more **pseudo-tool description blocks** -- structured text segments delimited by special begin/end tokens -- the system interprets these as implicit retrieval queries. Each pseudo-tool description is a natural language specification of a desired API capability (e.g., "a function that retrieves current weather data for a given city").

### 4.2 Retrieval Model and Corpus

The retrieval module encodes queries and corpus entries into a shared embedding space using a sentence-level encoder (SimCSE, based on RoBERTa-large). The corpus consists of natural language descriptions of all available APIs, structured as a mapping from description text to a composite key of the form `category[SEP]tool_name[SEP]api_name`.

At initialization, the entire corpus is pre-encoded into dense vectors. At query time, the pseudo-tool description is encoded and compared against the corpus embeddings via cosine similarity. The top-k most similar entries are returned, along with their similarity scores.

The retriever supports two deployment modes:
- **Local**: The embedding model and corpus are loaded in-process.
- **Remote**: A shared retriever server accepts HTTP requests, enabling multiple evaluation workers to share a single GPU-resident model.

### 4.3 Tool Schema Injection

Retrieved API entries are resolved to their full specifications (parameter schemas, endpoint metadata) and converted into the LLM's function-calling format. These are appended to the node's function schema, enabling the LLM to invoke them in subsequent reasoning steps. Deduplication ensures that APIs already present in the schema are not added again.

### 4.4 Duplicate Suppression via Tool Memory

Before triggering retrieval, the system compares the proposed pseudo-tool descriptions against a **tool memory** -- a shared data structure that records all pseudo-tool intents from prior search branches. The comparison uses token-level sequence similarity (SequenceMatcher). If the similarity exceeds a threshold (default: 0.82), or one description is a substring of the other, the retrieval is skipped and an informational message is injected instead. This mechanism prevents redundant retrieval across sibling branches in the search tree.

---

## 5. LLM Integration

### 5.1 Function Calling Protocol

The system communicates with the LLM via an OpenAI-compatible chat completions API. Tool schemas are passed as a `tools` parameter in the API request, and the LLM may respond with one or more `tool_calls` entries specifying function names and JSON-encoded arguments.

### 5.2 Message Construction

The conversation history presented to the LLM at each node consists of:

1. A **system message** containing the task preamble, which describes the agent's role and available tool categories.
2. A **user message** containing the natural language query.
3. The accumulated sequence of assistant messages (thoughts, function calls) and tool response messages (observations) along the path from root to the current node.
4. Optional injected messages: diversity prompts (when regenerating siblings), retrieval summary messages (after dynamic retrieval), and end-of-budget prompts (when approaching the query limit).

Messages flagged as invalid (e.g., consumed diversity prompts) are filtered out before submission to the LLM.

### 5.3 Observation Truncation

API observations may be arbitrarily long. The system truncates observations to a configurable maximum length (default: 1,024 characters) to prevent token budget exhaustion. Additional compression strategies (filtering by response schema, random subsampling) are available but truncation is the default.

### 5.4 Token Management and Retries

The system estimates input token counts and caps completion tokens to stay within the model's context window. For local models served via inference engines, the completion budget is dynamically computed as `context_length - estimated_input - buffer`. Requests are retried with exponential backoff (up to 3 attempts) on transient failures, with a secondary retry loop (up to 6 attempts) at the application level.

---

## 6. Meta-Strategy Layer

The meta-strategy layer sits above the base DFSDT algorithm and orchestrates how pseudo-tool descriptions are generated, refined, and selected before tools are ultimately retrieved and injected into the search tree.

### 6.1 Strategy Routing

When the LLM produces pseudo-tool description blocks during dynamic retrieval, the system routes the retrieval process through one of several strategies based on configuration:

| Strategy          | Description |
|-------------------|-------------|
| **Single Pass**   | Each pseudo-tool description is used as-is for a single retrieval query. No refinement. |
| **DBD (Description-by-Description)** | Each pseudo-tool description anchors an independent lineage. Over multiple turns, the description is iteratively refined using retrieved tool exemplars as feedback, producing progressively better retrieval queries. |
| **Scattershot**   | For each ancestor pseudo-tool, multiple variant descriptions are generated in parallel by the LLM. Each variant triggers independent retrieval. A population-level voting mechanism aggregates results across variants. |
| **Genetic Algorithm (GA)** | A population of pseudo-tool descriptions is evolved over multiple generations using LLM-driven crossover and mutation operators. Fitness is based on retrieval similarity scores, with a penalty for overlap with previously explored intents. |
| **Memetic Algorithm (MA)** | Extends the genetic algorithm with a local search (refinement) phase: after crossover and mutation, each offspring is individually refined using retrieved tool exemplars before fitness evaluation. |

### 6.2 Population-Level Tool Selection

The evolutionary strategies (GA and MA) and Scattershot employ a **population-level voting** mechanism for final tool selection. Rather than selecting tools from a single best description, the system aggregates retrieval results across all individuals in the final population:

1. Each individual's top-k retrieved tools cast a vote for the corresponding API.
2. Votes are tallied across the population, with tie-breaking by average rank (ascending) and average similarity score (descending).
3. The top tools by vote count are selected, subject to a tool budget constraint.
4. Only tools that are verifiably present in the API corpus (i.e., appear in at least one resolved retrieval result) are retained.

### 6.3 Evolutionary Operators

The genetic operators are implemented as LLM calls:

- **Crossover**: Given two parent pseudo-tool descriptions and the original ancestor description, the LLM is prompted to combine complementary aspects of both parents into a single offspring.
- **Mutation**: Given a single parent, the LLM is prompted to introduce meaningful variation while preserving relevance to the original query.
- **Refinement** (memetic only): After genetic operators, each offspring is refined using the same prompt structure as the DBD strategy -- presenting retrieved tool exemplars and asking the LLM to improve the description.

### 6.4 Fitness Evaluation

Fitness is computed from retrieval quality minus memory overlap:

```
fitness = alpha * top1_similarity + (1 - alpha) * top3_mean_similarity - memory_penalty
```

where:
- `top1_similarity` is the best retrieval similarity score,
- `top3_mean_similarity` is the average of the top retrieved similarity scores,
- `memory_penalty` is the Jaccard similarity to the most similar prior intent, gated by the configured threshold.

---

## 7. Tool Memory and Cross-Pass Learning

### 7.1 Purpose

Tool memory is a shared data structure that persists across all branches of a single DFSDT search tree and, when used within a meta-strategy, across multiple evolutionary generations. Its purpose is to record the pseudo-tool intents that have been explored, enabling the system to suppress redundant retrieval and encourage diverse exploration.

### 7.2 Contents

Tool memory stores normalized pseudo-tool description strings as keys. Only the intent text is retained; the actual retrieved tool payloads are intentionally excluded. This design keeps the memory lightweight and focused on duplication suppression rather than caching retrieval results.

### 7.3 Injection and Consumption

Tool memory is populated at two points:

1. **After retrieval within DFSDT**: When a non-evolutionary strategy (single pass, DBD) completes retrieval, the pseudo-tool descriptions from the retrieval iterations are added to the shared memory dictionary.
2. **After evolutionary finalization**: When a genetic or memetic strategy completes, both the original ancestor description and the best-evolved description are recorded in memory.

Tool memory is consumed at two points:

1. **Before retrieval in DFSDT**: The duplicate suppression check (Section 4.4) compares new pseudo-tool blocks against memory entries. If a near-duplicate is detected, retrieval is skipped.
2. **During fitness evaluation**: The memory penalty component of the fitness function (Sections 6.4) discourages the evolution of descriptions too similar to previously explored intents.

### 7.4 Scope

Within a single DFSDT invocation, tool memory is initialized as an empty dictionary and accumulates entries as branches are explored. When multiple DFSDT passes are orchestrated by a meta-strategy, the same memory object is shared across passes, enabling cross-pass learning. This allows later passes to benefit from the exploration performed by earlier ones, progressively narrowing the search toward unexplored regions of the tool space.
