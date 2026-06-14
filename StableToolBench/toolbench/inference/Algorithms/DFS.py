import re
from difflib import SequenceMatcher
from textwrap import dedent
from Tree.Tree import my_tree, tree_node
from Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION, FORMAT_INSTRUCTIONS_USER_FUNCTION
from Prompts.Tree_search_prompts import DIVERSITY_PROMPT, END_PROMPT
from Algorithms.base_search import base_search_method
from copy import deepcopy
from LLM_rank.rank_candidate import sum_based_rankn, rank2_subfix
import json
import random
from toolbench.inference.LLM.tokens import *
from toolbench.inference.LLM.prompts import build_root_planner_messages, static_mode_preamble

class DFS_tree_search(base_search_method):

    def __init__(self, llm, io_func, process_id=0, callbacks=None, planning_llm=None):
        super(DFS_tree_search, self).__init__(
            llm, io_func, process_id, callbacks)
        """Depth-first search.
        with_filter=True: Every time a child node is generated, choose the best multiple iterations to go.
        with_filter=False: Do as Preorder traversal.
        """

        self.io_func = io_func #io_func=io_state=env is an instantiation of rapidapi_wrapper
        self.llm = llm # the llm is not a part of the environment but the tree
        self.planning_llm = planning_llm

        self.process_id = process_id
        # Add for dynamic retrieval
        self.retriever = io_func.retriever
        self.retrieve_mode = io_func.retrieve_mode
        self.io_func.retriever = None # Do not need to copy retriever when deepcopy the io_func -> set to None to prevent deepcopy of env.retriever
        self.restart() # initialize via reset design pattern defines what a "fresh" agent should look like

        self.callbacks = callbacks if callbacks is not None else []

    def restart(self):
        self.status = 0
        self.terminal_node = []
        self.give_up_node = []
        self.now_expand_num = 0
        self.query_count = 0
        self.total_tokens = 0
        self._planner_used = False
        # Shared across all expanded nodes in this search. The dictionary keys
        # collect ancestor pseudo-tool descriptions emitted by planning
        # strategies (scattershot, DBD, single-pass, etc.). Each DFS chain can
        # therefore see what other chains have already explored and avoid
        # redundant retrieval work.
        self.tool_memory: dict = {}

    def send_agent_chain_end(self, depth, agent_block_ids, chain_block_ids):
        for i in range(len(self.callbacks)):
            callback = self.callbacks[i]
            callback.on_chain_end(
                depth=depth,
                block_id=chain_block_ids[i]
            )
            if i < len(agent_block_ids):
                callback.on_agent_end(
                    depth=depth,
                    block_id=agent_block_ids[i]
                )

    def to_json(self, answer=False, process=True):

        if process:
            json_obj = {
                "win": self.status == 1,
                "tree": self.tree.to_json_recursive(),
                "forward_args": self.forward_args,
                "compare_candidates": [],
            }
            for node in self.terminal_node:
                if node.pruned == False:  # has answer
                    json_obj["compare_candidates"].append(
                        node.get_chain_result_from_this_node(use_messages=False))
        else:
            json_obj = {}

        if answer:
            json_obj["answer_generation"] = {
                "valid_data": False,
                "query_count": self.query_count,
                "total_tokens": self.total_tokens,
                "final_answer": "",
                "finish_type": "give_answer",
                "function": self.io_func.functions,
                "chain": [],
            }
            for node in self.terminal_node:
                if node.pruned == False:
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_answer"
                    json_obj["answer_generation"]["final_answer"] = node.description
                    json_obj["answer_generation"]["train_messages"] = node.get_train_messages_from_this_node(
                    )
                    break
            # do not have final answer, look for give_up
            if json_obj["answer_generation"]["valid_data"] == False:
                if len(self.give_up_node) > 0:
                    random_pos = random.randint(0, len(self.give_up_node) - 1)
                    choose_give_up_node = self.give_up_node[random_pos]
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_up"
                    json_obj["answer_generation"]["final_answer"] = choose_give_up_node.description
                    json_obj["answer_generation"]["train_messages"] = choose_give_up_node.get_train_messages_from_this_node()
        return json_obj
    
    @staticmethod
    def _has_payload(msg: dict) -> bool:
        """
        A message is 'usable' if it has non-empty text content or at least one tool_call.
        Kept for backward-compat; DFSDT progress requires _has_actionable_payload.
        """
        if not isinstance(msg, dict):
            return False
        text = (msg.get("content") or "").strip()
        return bool(text) or (bool(msg.get("tool_calls")) and len(msg["tool_calls"]) > 0)

    @staticmethod
    def _has_actionable_payload(msg: dict) -> bool:
        """
        A message ADVANCES DFSDT only if it contains either:
        (a) at least one tool_call (the agent invokes a tool / Finish / retrieval), OR
        (b) a `<|begin_func_description|>...<|end_func_description|>` block in the
            content text (the memetic-retrieval trigger pattern).

        Non-actionable PROSE — "Let me think about this..." or an empty
        explanation with no directive — does NOT advance the tree; recursing
        on a node holding only prose re-issues the same prompt and the model
        produces more prose, exhausting `max_query_count` (the canonical
        qc=105 dead-loop).

        This is a STRICTER predicate than `_has_payload`: `_has_payload=True
        and _has_actionable=False` means "the assistant said something but
        didn't direct any action" — that's a dead chain, prune it.
        """
        if not isinstance(msg, dict):
            return False
        if msg.get("tool_calls") and len(msg["tool_calls"]) > 0:
            return True
        text = (msg.get("content") or "")
        return has_func_description(text)

    def _compare_with_tool_memory(self, message_text: str):
        """
        Run a stateless LLM call to compare the latest pseudo-tools against the
        shared tool memory. This avoids relying on the node's conversation
        history and helps reduce redundant retrievals.

        The chain of events is:
        1. The current assistant message yields new pseudo-tool blocks.
        2. Before retrieval begins, we compare those blocks with tool_memory,
           which contains pseudo-tools and resolved tool descriptions gathered
           from every other expanded chain.
        3. A lightweight, stateless LLM call performs the similarity check so
           we do not depend on the node's message history.
        4. If the planner decides there is overlap, it can short-circuit or
           adjust retrieval, reducing duplicated work downstream.

        Returns a decision dictionary with keys:
        - is_duplicate (bool)
        - matched_keys (list[str])
        - summary (str)
        """
        decision = {
            "is_duplicate": False,
            "matched_keys": [],
            "summary": "",
        }

        if not self.tool_memory:
            decision["summary"] = "No prior pseudo-tools available for comparison."
            return decision

        pseudo_blocks = [b.strip() for b in FUNC_DESC_PATTERN.findall(message_text or "") if b and b.strip()]
        if not pseudo_blocks:
            decision["summary"] = "No pseudo-tool blocks were detected in the latest message."
            return decision

        matches = set()
        for candidate in pseudo_blocks:
            candidate_lower = candidate.lower()
            for ancestor_key in self.tool_memory:
                if not ancestor_key:
                    continue
                ancestor_lower = str(ancestor_key).lower()
                similarity = SequenceMatcher(None, candidate_lower, ancestor_lower).ratio()
                if (
                    similarity >= 0.82
                    or candidate_lower in ancestor_lower
                    or ancestor_lower in candidate_lower
                ):
                    matches.add(str(ancestor_key))

        if matches:
            decision["is_duplicate"] = True
            decision["matched_keys"] = sorted(matches)
            sample_keys = ", ".join(decision["matched_keys"][:3])
            more_flag = "" if len(decision["matched_keys"]) <= 3 else ", ..."
            decision["summary"] = (
                f"The proposed pseudo-tools overlap with existing entries such as: {sample_keys}{more_flag}. "
                "Avoid redundant retrieval unless you are pursuing a genuinely new angle."
            )
        else:
            decision["summary"] = "New pseudo-tools look distinct from stored entries; retrieval can proceed."

        return decision

    def start(self, single_chain_max_step, tree_beam_size, max_query_count, answer=1, with_filter=True):
        """ single_chain_max_step: The maximum depth of the tree
            tree_beam_size: How many children nodes for one node are generated per layer
            answer = n means the Algo exits when find n "give_answer" nodes
            max_query_count: the Algo exits when OpenAI-query exists this value
            with_filter: This is the difference between normal DFS(with_filter=True) and DFSDT(with_filter=False). 
        """
        self.forward_args = deepcopy(locals())
        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")
        self.tree = my_tree()
        self.tree.root.node_type = "Action Input"
        self.tree.root.io_state = deepcopy(self.io_func)

        system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
        system = system.replace("{task_description}",
                                self.io_func.task_description)
        self.tree.root.messages.append({"role": "system", "content": system})

        user = FORMAT_INSTRUCTIONS_USER_FUNCTION
        user = user.replace("{input_description}",
                            self.io_func.input_description)
        self.tree.root.messages.append({"role": "user", "content": user})

        return self.DFS(self.tree.root, single_chain_max_step, tree_beam_size, max_query_count, answer, with_filter)

    def DFS(self, now_node, single_chain_max_step, tree_beam_size, max_query_count, answer, with_filter=True):
        """
        Returns the number of grids to go back. When a child node of a node generates a final answer or give up, it should go back a few more grids
        In a sense, the larger this value is, the more diverse it is, and it is GreedySearch@n when it is enlarged to infinity.
        """

        # this two value declares the rate to go back, Algo degrades to CoT when the value=Inf
        final_answer_back_length = 2
        prune_back_length = 2

        now_node.expand_num = self.now_expand_num
        self.now_expand_num += 1
        if now_node.get_depth() >= single_chain_max_step or now_node.pruned or now_node.is_terminal:
            if now_node.is_terminal:  # final answer
                self.status = 1
                self.terminal_node.append(now_node)
                return final_answer_back_length
            else:
                now_node.pruned = True
                if now_node.observation_code == 4:
                    self.give_up_node.append(now_node)
                    return prune_back_length
                else:
                    return 1

        next_tree_split_nodes = []
        for i in range(tree_beam_size):
            temp_now_node = now_node

            """If a node have children now, We will prompt the model to generate different nodes than all the existing nodes"""
            delete_former_diversity_message = False
            diversity_message = None
            if len(temp_now_node.children) > 0:

                former_candidates_des = ""
                js_list = []
                for k, child in enumerate(temp_now_node.children):
                    temp_node = child
                    while not temp_node.is_terminal and temp_node.node_type != "Action Input" and len(temp_node.children) > 0:
                        temp_node = temp_node.children[0]
                    if temp_node.node_type == "Action Input":
                        obj_dict = {
                            "name": temp_node.father.description,
                            "arguments": temp_node.description,
                            "function_output": temp_node.observation,
                            "mento-carlo-action-value": temp_node.compute_weight(),
                        }
                        js_list.append(obj_dict)

                if len(js_list) > 0:
                    former_candidates_des = former_candidates_des + \
                        f"{json.dumps(js_list,indent=2)}\n"
                    if temp_now_node.observation != "":
                        former_candidates_des = former_candidates_des + \
                            f"again, your former observation: {temp_now_node.observation}\n"
                    diverse_prompt = DIVERSITY_PROMPT
                    diverse_prompt = diverse_prompt.replace(
                        "{previous_candidate}", former_candidates_des)
                    diversity_message = {
                        "role": "user", "content": diverse_prompt}
                    temp_now_node.messages.append(diversity_message)

                    delete_former_diversity_message = True

            if self.query_count >= max_query_count and single_chain_max_step - now_node.get_depth() <= 3:
                end_messsgae = {
                    "role": "user", "content": END_PROMPT
                }
                temp_now_node.messages.append(end_messsgae)
            # on_chain_start
            now_depth = temp_now_node.get_depth() // 3
            chain_block_ids = [callback.on_chain_start(
                depth=now_depth,
                inputs=temp_now_node.messages
            ) for callback in self.callbacks]
            agent_block_ids = []
            self.llm.change_messages(temp_now_node.messages)
            # on_llm_start
            [callback.on_llm_start(
                depth=now_depth,
                messages=temp_now_node.messages
            ) for callback in self.callbacks]

                        # ---------- Planner gate (root-only; accept only usable payload) ----------
            planner_result = None
            planner_error = 0
            planner_tokens = 0
            use_planner = False

            if (
                self.planning_llm is not None
                and not getattr(self, "_planner_used", False)
                and temp_now_node is self.tree.root
            ):
                planner_messages = build_root_planner_messages(
                    query=self.io_func.input_description,
                    task_description=self.io_func.task_description,
                )
                r, e, tk = self.planning_llm.parse_with_messages(
                    messages=planner_messages,
                    tools=[],  # planner step: no tools
                    process_id=self.process_id,
                )
                # Use the stricter actionable
                # predicate so a prose-only planner result falls through to
                # the primary LLM instead of being adopted as `new_message`
                # and then pruned downstream by the actionable-only fix at
                # line 587 (which would kill the root and skip primary LLM
                # entirely). Planner output is only useful if it directs an
                # action (tool_call or `<|begin_func|>` block).
                if e == 0 and r is not None and DFS_tree_search._has_actionable_payload(r):
                    planner_result, planner_error, planner_tokens = r, e, tk
                    use_planner = True
                    self._planner_used = True

            # Produce the assistant message (planner if usable; else primary LLM)
            if use_planner:
                new_message = {k: v for k, v in planner_result.items() if v is not None}
                new_message.setdefault("role", "assistant")
                error_code = 0
                total_tokens = planner_tokens
                source = "planner"
                
            else:
                # 2026-05-25: ``rapidapi_wrapper`` defers appending Finish
                # until after the first retrieval (dynamic mode only). So
                # before any retrieval has happened ``functions`` is empty
                # and the model is forced into pure-chat mode → emits the
                # BEGIN/END description block per the system prompt →
                # memetic dispatches → Finish appears in the toolset on
                # subsequent turns. No per-turn filtering needed here.
                new_message, error_code, total_tokens = self.llm.parse(
                    temp_now_node.io_state.functions, process_id=self.process_id,
                )
                source = "primary"

            new_message = {k:v for k,v in new_message.items() if v != None}
            # on_llm_end
            [callback.on_llm_end(
                depth=now_depth,
                response=new_message
            ) for callback in self.callbacks]
            self.query_count += 1
            self.total_tokens += total_tokens
                
            # Force quit
            if self.query_count >= max_query_count+5:
                return 100000

            # We need to exclude the diversity_message, because it will influence child nodes
            if delete_former_diversity_message:
                temp_now_node.messages[-1]["valid"] = False

            # parse nodes from OpenAI-message like CoT method
            assert new_message["role"] == "assistant"

            # 2026-05-25: detect the retrieval-protocol intent loosely. Newer
            # tool-eager reasoning models sometimes emit BEGIN without
            # END and a simultaneous tool_call in the same turn. We still
            # route to memetic dispatch as long as the agent wrote BEGIN —
            # the strict pattern stays the source of truth for *extraction*
            # of clean pseudo-tool blocks (handled inside memetic via
            # `extract_func_descriptions(text, tolerant=True)` where needed).
            content_text = new_message.get("content") or ""
            retrieval_dispatched = False
            # Root-node ablation (--disable-midexec-retrieval): honour the
            # retrieval trigger ONLY for the first retrieval on this lineage
            # (the root retrieval) and suppress every subsequent (mid-execution)
            # trigger. In dynamic mode the wrapper starts with zero tools
            # (rapidapi.py:191-192), so the first has_func_description trigger
            # IS the root retrieval; that single dynamic_retrieve_base_on_des
            # call internally runs the full dbd multi-turn refine loop, so
            # "root-only" still gets the db-turn refinement. `_retrieval_done`
            # is set on child_io_state after the first retrieval and inherited
            # by descendants via deepcopy, so deeper nodes are gated off.
            _disable_midexec = bool(getattr(self.io_func, "disable_midexec_retrieval", False))
            _already_retrieved = bool(getattr(temp_now_node.io_state, "_retrieval_done", False))
            _block_retrieval = _disable_midexec and _already_retrieved
            if has_func_description(content_text) and not _block_retrieval:
                temp_node = tree_node()
                temp_node.node_type = "Thought"
                temp_node.description = content_text
                child_io_state = deepcopy(temp_now_node.io_state)
                child_io_state.retriever=None
                retrieval_summary_message = None
                retrieval_decision = {
                    "is_duplicate": False,
                    "matched_keys": [],
                    "summary": "",
                }

                if self.retrieve_mode == "dynamic":  # dynamic multi-turn retrieval
                    retrieval_decision = self._compare_with_tool_memory(new_message["content"])
                    if retrieval_decision.get("is_duplicate"):
                        overlap_notice = {
                            "role": "system",
                            "content": (
                                "This pseudotool substantially overlaps with an existing ancestor pseudotool. "
                                "Avoid redundant retrieval unless you are exploring a genuinely new angle."
                            ),
                        }
                        temp_now_node.messages.append(overlap_notice)
                        retrieval_iterations = []
                        retrieval_summary_message = {
                            "role": "system",
                            "content": retrieval_decision.get("summary", "Retrieval skipped due to overlap."),
                        }
                        child_io_state.last_retrieval_message = retrieval_summary_message
                        child_io_state.last_retrieval_details = {
                            "strategy": "skipped_due_to_duplicate_pseudotool",
                            "matched_keys": retrieval_decision.get("matched_keys", []),
                        }
                    else:
                        retrieval_iterations = child_io_state.dynamic_retrieve_base_on_des(
                            new_message["content"], self.retriever, self.llm, tool_memory=self.tool_memory
                        )
                        # Root-node ablation bookkeeping: mark this lineage as
                        # having performed its (root) retrieval. Descendants
                        # inherit this via deepcopy of child_io_state, so any
                        # further has_func_description trigger is gated off when
                        # --disable-midexec-retrieval is set. No-op otherwise.
                        child_io_state._retrieval_done = True
                        retrieval_summary_message = getattr(child_io_state, "last_retrieval_message", None)
                        # calling sanity_check code
                        self.io_func.write_sanity_check(
                            query_id=getattr(child_io_state, "query_id", None),
                            method=getattr(child_io_state, "method", None),
                            query=getattr(child_io_state, "input_description", None),
                            retrieval_iterations=retrieval_iterations,
                            instruction_file=getattr(child_io_state, "input_query_file", None),
                            retrieval_idx=self.io_func.sanity_check_idx,
                        )
                        self.io_func.sanity_check_idx += 1

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = deepcopy(temp_now_node.messages) # prior messages are kept
                if self.retrieve_mode == "dynamic":
                    if retrieval_summary_message is not None:
                        temp_node.messages.append(retrieval_summary_message)
                    if _disable_midexec and getattr(child_io_state, "_retrieval_done", False):
                        # Root-node ablation: the single root retrieval is done.
                        # Switch the agent OUT of dynamic-retrieval mode by
                        # rewriting the system message (index 0) to the STATIC
                        # tool-use preamble listing the tools just retrieved. The
                        # agent then behaves as a standard function-caller and is
                        # genuinely unaware of the <|begin_func_description|>
                        # mid-execution retrieval feature for the rest of the
                        # trajectory — so it never re-emits it (which would be
                        # gated off, leaving the agent to loop until
                        # max_query_count). The dynamic preamble was only needed
                        # to trigger THIS root retrieval. Propagates to all
                        # descendants via the message deepcopy at child creation.
                        _tool_lines = []
                        for _k, _fn in enumerate(getattr(child_io_state, "functions", []) or [], 1):
                            _f = _fn.get("function", {}) if isinstance(_fn, dict) else {}
                            _name = _f.get("name", "")
                            _desc = ((_f.get("description") or "")[:512].replace("\n", "").strip()) or "None"
                            _tool_lines.append(f"{_k}.{_name}: {_desc}")
                        _static_sys = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION.replace(
                            "{task_description}", static_mode_preamble(_tool_lines)
                        )
                        if temp_node.messages and temp_node.messages[0].get("role") == "system":
                            temp_node.messages[0] = {"role": "system", "content": _static_sys}
                    self.llm.change_messages(temp_node.messages)
                temp_node.father = temp_now_node
                temp_now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                temp_now_node = temp_node

                if error_code != 0:
                    temp_now_node.observation_code = error_code
                    temp_now_node.pruned = True

                # Mark retrieval as having been dispatched this turn — see
                # `retrieval_dispatched` guard on the tool_calls block below.
                # If the agent emitted both a description AND a tool call
                # (a retrieval-protocol violation), the description
                # wins and we drop the tool call: the agent was supposed to
                # WAIT for retrieval, not call anything.
                retrieval_dispatched = True

            # if "function_call" in new_message.keys():
            if (
                "tool_calls" in new_message.keys()
                and new_message["tool_calls"] is not None
                and len(new_message["tool_calls"]) > 0
                and not retrieval_dispatched   # 2026-05-25: see comment above
            ):
                tool_calls = new_message["tool_calls"]
                if self.process_id == 0:
                    print("number of parallel calls:",len(tool_calls))

                for i in range(len(tool_calls)):
                # on_agent_action
                    agent_block_ids = [callback.on_agent_action(
                        depth=now_depth,
                        # action=new_message["function_call"]["name"],
                        action=tool_calls[i]["function"]["name"],
                        # action_input=new_message["function_call"]["arguments"]
                        action_input=tool_calls[i]["function"]["arguments"]
                    ) for callback in self.callbacks]
                    # function_name = new_message["function_call"]["name"]
                    function_name = tool_calls[i]["function"]["name"]
                    temp_node = tree_node()
                    temp_node.node_type = "Action"
                    temp_node.description = function_name
                    child_io_state = deepcopy(temp_now_node.io_state)
                    child_io_state.retriever=None

                    temp_node.io_state = child_io_state
                    temp_node.is_terminal = child_io_state.check_success() != 0
                    temp_node.messages = deepcopy(temp_now_node.messages)
                    temp_node.father = temp_now_node
                    temp_now_node.children.append(temp_node)

                    temp_node.print(self.process_id)
                    temp_now_node = temp_node

                    # function_input = new_message["function_call"]["arguments"]
                    function_input = tool_calls[i]["function"]["arguments"]
                    temp_node = tree_node()
                    temp_node.node_type = "Action Input"
                    temp_node.description = function_input
                    child_io_state = deepcopy(temp_now_node.io_state)
                    child_io_state.retriever=None
                    
                    # on_tool_start
                    [callback.on_tool_start(
                        depth=now_depth,
                        tool_name=temp_now_node.description,
                        tool_input=function_input
                    ) for callback in self.callbacks]
                    observation, status = child_io_state.step(
                        action_name=temp_now_node.description, action_input=function_input)
                    temp_node.observation = observation
                    temp_node.observation_code = status

                    temp_node.io_state = child_io_state
                    temp_node.is_terminal = child_io_state.check_success() != 0
                    temp_node.messages = deepcopy(temp_now_node.messages)
                    temp_node.father = temp_now_node
                    temp_now_node.children.append(temp_node)
                    temp_node.print(self.process_id)
                    temp_now_node = temp_node
                    # on_tool_end
                    [callback.on_tool_end(
                        depth=now_depth,
                        output=observation,
                        status=status
                    ) for callback in self.callbacks]
                    if status != 0:
                        # return code defination can be seen in Downstream_tasks/rapid_api
                        if status == 4:
                            temp_now_node.pruned = True
                        elif status == 1:  # hallucination api name
                            # assert "function_call" in new_message.keys()
                            # new_message["function_call"]["name"] = "invalid_hallucination_function_name"
                            assert "tool_calls" in new_message.keys() and len(new_message["tool_calls"]) > 0
                            tool_calls[i]["function"]["name"] = "invalid_hallucination_function_name"
                        elif status == 3:  # final answer
                            temp_now_node.is_terminal = True
                            # Store all the retrieved functions
                            self.io_func.functions = temp_now_node.io_state.functions
                            temp_now_node.make_finish(final_answer_back_length)
                    if i == 0:
                        temp_now_node.messages.append(new_message)
                    if temp_now_node.node_type == "Action Input":
                        temp_now_node.messages.append({
                            # "role": "function",
                            # "name": new_message["function_call"]["name"],
                            # "content": temp_now_node.observation,
                            "role":"tool",
                            # "name": new_message["function_call"]["name"],
                            "name": tool_calls[i]["function"]["name"],
                            "content": temp_now_node.observation,
                            "tool_call_id": tool_calls[i]['id'],
                        })
            else:
                # This else-branch is reached in the no-tool-call,
                # no-retrieval-dispatch path. Two failure modes both end here:
                #   (1) empty payload (no content, no tool_calls) — the
                #       original 105-call dead-loop on broken Responses input.
                #   (2) non-actionable PROSE ("Let me think...", error text,
                #       give-up wording without a Finish call) — DFS accepting
                #       non-actionable text as "progress" is the remaining
                #       qc=105 root cause beyond the converter fix.
                # In BOTH cases recursing re-issues the same prompt and burns
                # max_query_count for nothing.
                #
                # Carve-out: only short-circuit
                # in the DFSDT path (with_filter=False) where backtrack is
                # clean. For with_filter=True (filtered DFS / beam-rank path),
                # a non-actionable LATER beam sample must not prune the parent
                # because earlier actionable candidates are still queued for
                # sum_based_rankn. There we just append + continue — the rank
                # step will downweight the bad candidate.
                if (not with_filter) and not self._has_actionable_payload(new_message):
                    if self.process_id == 0:
                        import sys as _sys
                        text = (new_message.get("content") or "")[:200]
                        n_tc = len(new_message.get("tool_calls") or [])
                        print(f"[DFSDT-SALVAGE] depth={now_depth} tc={n_tc} chars={len(new_message.get('content') or '')} text={text!r}", file=_sys.stderr, flush=True)
                    # Still append the message so the parent chain has a
                    # complete record of what the agent emitted, without
                    # affecting downstream DFS progress.
                    temp_now_node.messages.append(new_message)
                    temp_now_node.pruned = True
                    # Without this enrollment, to_json (DFS.py:82-109) finds both
                    # terminal_node and give_up_node empty when every chain in
                    # the run hits non-actionable pruning. Result:
                    # valid_data=False, chain=[], final_answer="", finish_type=
                    # "give_answer" (the LIE). A previous attempt set
                    # observation_code=4 but the parent recursion at line 250-261
                    # doesn't re-enter on this pruned node, so enrollment never
                    # fires. The correct fix
                    # is to enroll the node DIRECTLY here so to_json's salvage
                    # path finds it → produces valid_data=True, finish_type=
                    # "give_up", final_answer=description with truthful marker.
                    temp_now_node.observation_code = 4
                    temp_now_node.description = (
                        "[empty-reasoning-output] model returned no actionable "
                        f"content (content_chars={len(new_message.get('content') or '')}, "
                        "tool_calls=0); salvaged as give_up so trajectory is "
                        "preserved (judge will mark as Unsolved, but trajectory is valid)."
                    )
                    self.give_up_node.append(temp_now_node)
                    self.send_agent_chain_end(
                        now_depth, agent_block_ids, chain_block_ids)
                    return prune_back_length
                temp_now_node.messages.append(new_message)
            return_value = None
            if not with_filter:  # DFSDT
                result = self.DFS(temp_now_node, single_chain_max_step,
                                  tree_beam_size, max_query_count, answer, with_filter)
                if len(self.terminal_node) >= answer:
                    return_value = 10000
                elif result > 1:
                    return_value = result-1

            else:

                next_tree_split_nodes.append(temp_now_node)
            self.send_agent_chain_end(
                now_depth, agent_block_ids, chain_block_ids)
            if return_value is not None:
                return return_value

        # Sort the generated next_tree_split_nodes nodes when normal DFS
        if len(next_tree_split_nodes) > 1:
            # When using normal DFS, if we have 
            # many child nodes, we will refer to LLM to compare and choose the best one to expand first
            # remember, this operator will cost extra OpenAI calls.
            LLM_rank_args = {
                "functions": self.io_func.functions,
                "process_id": self.process_id,
                "task_description": self.io_func.task_description,
                "rank_func": rank2_subfix,
            }
            scores, rank_query_count, total_tokens = sum_based_rankn(
                self.llm, LLM_rank_args=LLM_rank_args, candidates=next_tree_split_nodes)
            self.query_count += rank_query_count
            self.total_tokens += total_tokens
            for score, node in zip(scores, next_tree_split_nodes):
                node.prior_score = score
            zip_value = list(
                zip(next_tree_split_nodes, range(len(next_tree_split_nodes))))
            zip_value.sort(
                key=lambda x: x[0].prior_score, reverse=True)  # 先做score高的
            next_tree_split_nodes, filtered_order = zip(*zip_value)
            # if self.process_id == 0:
            #     print(f"score={scores}, filtered order: {filtered_order}")

        '''
        Choose one to expand
        '''
        for i in range(len(next_tree_split_nodes)):
            result = self.DFS(
                next_tree_split_nodes[i], single_chain_max_step, tree_beam_size, max_query_count, answer)
            if len(self.terminal_node) >= answer:
                return 10000
            elif result > 1:
                now_node.make_finish(2)
                return result - 1

        return 1
