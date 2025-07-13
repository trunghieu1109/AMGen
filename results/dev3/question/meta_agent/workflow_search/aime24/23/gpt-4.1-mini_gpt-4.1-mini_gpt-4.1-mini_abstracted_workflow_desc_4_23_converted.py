async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Define variables a,b,c,d,e,f as digits in the 2x3 grid (top row: a,b,c; bottom row: d,e,f), each 0-9. "
        "Formulate the two main sum constraints as linear equations: (100a + 10b + c) + (100d + 10e + f) = 999 and (10a + d) + (10b + e) + (10c + f) = 99. "
        "Validate these equations using the example grid [[0,0,8],[9,9,1]] where 008 + 991 = 999 and 09 + 09 + 81 = 99. "
        "Explicitly confirm assumptions about digit ranges, allowing leading zeros and repeated digits by analyzing the example and problem context. "
        "If ambiguity remains, plan to enumerate solutions under both assumptions later."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, defining variables and validating constraints, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_1 = (
        "Sub-task 2: Introduce carry variables carry1 and carry2 for digit-wise addition of the two 3-digit row numbers summing to 999. "
        "Formulate digit-wise addition constraints incorporating carry variables: units column: c + f = 9 + 10 * carry1; "
        "tens column: b + e + carry1 = 9 + 10 * carry2; hundreds column: a + d + carry2 = 9. "
        "Ensure these constraints are consistent with the overall row sum equation. "
        "Avoid ignoring carry logic or assuming digit sums equal 9 without carry consideration."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_1,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "CoT"
    }
    thinking_1, answer_1 = await cot_agent_1([taskInfo, thinking_0, answer_0], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, introducing carry variables and formulating digit-wise addition constraints, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 3: Incorporate the column sum constraint (sum of three 2-digit column numbers equals 99) alongside carry constraints. "
        "Express column sums as (10a + d) + (10b + e) + (10c + f) = 99 and verify consistency with carry-based digit-wise addition. "
        "Analyze how these constraints interact and restrict digit and carry variables. "
        "Avoid treating constraints independently or ignoring their combined effect."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking_1, answer_1],
        "agent_collaboration": "CoT"
    }
    thinking_2, answer_2 = await cot_agent_2([taskInfo, thinking_1, answer_1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, incorporating column sum constraint and analyzing combined restrictions, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 4: Enumerate all possible digit tuples (a,b,c,d,e,f) and carry values (carry1, carry2) within their domains (digits 0-9, carries 0 or 1) that satisfy all digit-wise addition and column sum constraints. "
        "Use constraint propagation, pruning, or systematic search to efficiently explore the solution space without brute force enumeration of all 10^6 possibilities. "
        "Ensure all constraints, including digit bounds, carry logic, and sum equations, are simultaneously respected. "
        "Avoid incomplete enumeration or ignoring carry constraints to prevent undercounting solutions. Provide the count of all valid solutions."
    )
    N_sc = self.max_sc
    cot_sc_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinking_map_3 = {}
    answer_map_3 = {}
    subtask_desc_3 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking_2, answer_2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_3[i]([taskInfo, thinking_2, answer_2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3[i].id}, enumerating valid digit and carry tuples, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3.append(answer_i.content)
        thinking_map_3[answer_i.content] = thinking_i
        answer_map_3[answer_i.content] = answer_i
    best_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking_3 = thinking_map_3[best_answer_3]
    answer_3 = answer_map_3[best_answer_3]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 5: Analyze and simplify numeric relationships between digits and carry variables based on enumerated solutions. "
        "Identify patterns, parametric forms, or variable dependencies to reduce complexity or explain solution structure. "
        "Use insights to verify completeness of enumeration and potentially generalize counting. "
        "Avoid invalid assumptions or loss of generality during simplification."
    )
    cot_sc_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    thinking_map_4 = {}
    answer_map_4 = {}
    subtask_desc_4 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking_3, answer_3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_4[i]([taskInfo, thinking_3, answer_3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_4[i].id}, analyzing and simplifying digit and carry relations, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_4.append(answer_i.content)
        thinking_map_4[answer_i.content] = thinking_i
        answer_map_4[answer_i.content] = answer_i
    best_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking_4 = thinking_map_4[best_answer_4]
    answer_4 = answer_map_4[best_answer_4]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 5: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 6: Aggregate all valid digit assignments and carry values from enumeration and analysis to compute total number of distinct digit placements in the 2x3 grid satisfying both sum conditions. "
        "Cross-check final count against constraints and example grid to verify correctness. "
        "Provide detailed verification summary explaining carry handling and assumption validations to ensure completeness. "
        "Consider opinions from multiple debate agents to ensure correctness."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking_4, answer_4],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking_5, answer_5 = await agent([taskInfo, thinking_4, answer_4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking_4, answer_4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking_5, answer_5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, aggregating and verifying final count, thinking: {thinking_5.content}; answer: {answer_5.content}")
            all_thinking_5[r].append(thinking_5)
            all_answer_5[r].append(answer_5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo] + all_thinking_5[-1] + all_answer_5[-1], "Sub-task 6: Final aggregation and verification." + debate_instruction_5, is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing count, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 6: ", sub_tasks[-1])

    cot_reflect_instruction_6 = (
        "Sub-task 7: Conduct a Reflexion phase to critically review the entire reasoning and solution process, focusing on correctness of carry modeling, assumption validation (leading zeros and digit repetition), and completeness of enumeration. "
        "Address any discrepancies or overlooked cases identified during verification. "
        "If necessary, refine previous subtasks or final answer accordingly to ensure robustness and confidence in the final solution."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking_5, answer_5]
    subtask_desc_6 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", thinking_5, answer_5],
        "agent_collaboration": "Reflexion"
    }
    thinking_6, answer_6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, reviewing entire solution process, thinking: {thinking_6.content}; answer: {answer_6.content}")
    for i in range(N_max_6):
        feedback_6, correct_6 = await critic_agent_6([taskInfo, thinking_6, answer_6], "Please review and provide limitations of the solution. If correct, output exactly 'True'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback: {feedback_6.content}; correctness: {correct_6.content}")
        if correct_6.content == "True":
            break
        cot_inputs_6.extend([thinking_6, answer_6, feedback_6])
        thinking_6, answer_6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining solution after critique, thinking: {thinking_6.content}; answer: {answer_6.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking_6.content}; answer - {answer_6.content}")
    subtask_desc_6['response'] = {"thinking": thinking_6, "answer": answer_6}
    logs.append(subtask_desc_6)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_6, answer_6, sub_tasks, agents)
    return final_answer, logs
