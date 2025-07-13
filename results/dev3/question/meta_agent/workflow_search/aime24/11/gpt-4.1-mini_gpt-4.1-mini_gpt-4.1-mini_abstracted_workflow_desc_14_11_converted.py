async def forward_11(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Identify and verify the elements under constraints for the lattice paths problem. "
        "Confirm the definition of direction change as a switch between horizontal and vertical moves, "
        "clarify assumptions about starting direction, and establish that paths can be decomposed into five monotone segments alternating directions."
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
    agents.append(f"CoT agent {cot_agent_0.id}, analyzing problem elements, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_1 = (
        "Sub-task 2: Enumerate possible patterns of direction changes for the lattice paths problem. "
        "Consider both possible starting directions (horizontal first or vertical first) and confirm if both are allowed. "
        "Use self-consistency to ensure robustness of enumeration."
    )
    N_sc = self.max_sc
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    thinking_map_1 = {}
    answer_map_1 = {}
    subtask_desc_1 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before SC-CoT agents call: {subtask_desc_1}")
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, enumerating direction change patterns, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1.append(answer_i.content)
        thinking_map_1[answer_i.content] = thinking_i
        answer_map_1[answer_i.content] = answer_i
    best_answer_1 = Counter(possible_answers_1).most_common(1)[0][0]
    thinking_1 = thinking_map_1[best_answer_1]
    answer_1 = answer_map_1[best_answer_1]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 3: Derive a formal representation of the problem as sequences of segment lengths. "
        "Formulate constraints as compositions of 8 into positive parts depending on starting direction. "
        "Count the number of such compositions and ways to arrange steps within segments. "
        "Use self-consistency to verify the counting approach."
    )
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinking_map_2 = {}
    answer_map_2 = {}
    subtask_desc_2 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before SC-CoT agents call: {subtask_desc_2}")
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_2[i]([taskInfo, thinking_1, answer_1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, deriving formal representation, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2.append(answer_i.content)
        thinking_map_2[answer_i.content] = thinking_i
        answer_map_2[answer_i.content] = answer_i
    best_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking_2 = thinking_map_2[best_answer_2]
    answer_2 = answer_map_2[best_answer_2]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction_3 = (
        "Sub-task 4: Compute the number of valid paths with exactly four direction changes. "
        "For each possible starting direction, count compositions of 8 into required parts, multiply counts, and sum results. "
        "Use debate and reflexion to verify and finalize the result, cross-checking with known combinatorial identities or smaller cases."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds = self.max_round
    all_thinking_3 = [[] for _ in range(N_rounds)]
    all_answer_3 = [[] for _ in range(N_rounds)]
    subtask_desc_3 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_3,
        "context": ["user query", thinking_2.content, answer_2.content],
        "agent_collaboration": "Debate"
    }
    print(f"Logging before Debate agents call: {subtask_desc_3}")
    for r in range(N_rounds):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_2, answer_2], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2, answer_2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing final count, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_3[r].append(thinking_i)
            all_answer_3[r].append(answer_i)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3([taskInfo] + all_thinking_3[-1] + all_answer_3[-1], "Sub-task 4: Finalize and verify the count of paths with exactly four direction changes.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing count, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    return final_answer, logs
