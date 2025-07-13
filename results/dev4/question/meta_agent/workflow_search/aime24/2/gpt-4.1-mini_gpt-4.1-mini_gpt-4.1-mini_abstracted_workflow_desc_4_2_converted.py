async def forward_2(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Formally model the problem. Define the octagon vertices as positions 0 to 7. "
        "Define the coloring function from vertices to {red, blue} with independent probability 1/2 each. "
        "Describe the rotation group of order 8 acting on these vertices as rotations by multiples of 45 degrees. "
        "Formally express the condition that there exists a rotation g such that all blue vertices map onto vertices originally red. "
        "Output a structured JSON summary with keys: 'vertices', 'coloring_prob', 'rotation_group', 'condition'."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_0}")
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, formal modeling, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_1 = (
        "Sub-task 2: Using the formal model, explicitly compute the sizes |A_k| for each rotation k=0 to 7, "
        "where A_k is the set of colorings fixed by rotation k. Use cycle decomposition and combinatorial counting. "
        "Output a JSON object mapping each k to |A_k| as integers. Provide detailed numeric calculations in the reasoning."
    )
    N_sc = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    subtask_desc_1 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before SC-CoT agent calls: {subtask_desc_1}")
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, enumerating |A_k|, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1.append(answer_i.content)
        thinkingmapping_1[answer_i.content] = thinking_i
        answermapping_1[answer_i.content] = answer_i
    best_answer_1 = Counter(possible_answers_1).most_common(1)[0][0]
    thinking_1 = thinkingmapping_1[best_answer_1]
    answer_1 = answermapping_1[best_answer_1]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 3: Compute all pairwise intersections |A_i ∩ A_j| for i ≠ j, and higher-order intersections if needed, "
        "using cycle structures of combined rotations and combinatorial arguments. Output a JSON object with keys as tuples (i,j) mapping to intersection sizes. "
        "Provide detailed numeric calculations and reasoning."
    )
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc_2 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before SC-CoT agent calls: {subtask_desc_2}")
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_2[i]([taskInfo, thinking_1, answer_1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, enumerating intersections, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2.append(answer_i.content)
        thinkingmapping_2[answer_i.content] = thinking_i
        answermapping_2[answer_i.content] = answer_i
    best_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking_2 = thinkingmapping_2[best_answer_2]
    answer_2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 4: Using |A_k| and intersection sizes, perform inclusion-exclusion step-by-step to find the size of the union of all A_k. "
        "Calculate the probability as the ratio of favorable colorings to total colorings (256). Simplify the fraction m/n to lowest terms. "
        "Output JSON with fields: numerator m, denominator n, sum m_plus_n, and detailed intermediate steps. Verify fraction validity."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking_2.content, answer_2.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    print(f"Logging before CoT agent call: {subtask_desc_3}")
    thinking_3, answer_3 = await cot_agent_3([taskInfo, thinking_2, answer_2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, inclusion-exclusion and fraction simplification, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_4 = (
        "Sub-task 5: Verify the correctness and consistency of the simplified fraction and computed m+n. "
        "Cross-check against combinatorial identities and problem constraints. Provide justification or summary. "
        "If inconsistencies arise, request refinement; else finalize and output m+n with verification summary in JSON."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_4,
        "context": ["user query", thinking_3.content, answer_3.content],
        "agent_collaboration": "Debate | Reflexion"
    }
    print(f"Logging before Debate agent calls: {subtask_desc_4}")
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking_r, answer_r = await agent([taskInfo, thinking_3, answer_3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking_3, answer_3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking_r, answer_r = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying fraction and m+n, thinking: {thinking_r.content}; answer: {answer_r.content}")
            all_thinking_4[r].append(thinking_r)
            all_answer_4[r].append(answer_r)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 5: Final verification and answer output." + debate_instruction_4, is_sub_task=True)
    agents.append(f"Final Decision agent, final verification and answer, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_4, answer_4, sub_tasks, agents)
    return final_answer, logs
