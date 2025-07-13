async def forward_8(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: derive_and_validate_representations
    cot_instruction_1 = (
        "Sub-task 1: Derive a formal representation of the game states for the token game where a player can remove 1 or 4 tokens. "
        "Classify each state n (number of tokens) as winning or losing for the player to move. "
        "Establish base cases and recursive relations. Provide reasoning and a method suitable for n up to 2024."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_1: {subtask_desc_1}")
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, deriving game state classification, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Validate the classification of states derived in Sub-task 1 by computing winning/losing states for n=1 to 20. "
        "Confirm correctness and identify any pattern or periodicity. Use self-consistency by multiple agents to ensure robustness."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before subtask_2: {subtask_desc_2}")
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, validating states 1-20, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    best_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: select_and_verify_elements_under_constraints
    cot_sc_instruction_3 = (
        "Sub-task 3: Enumerate all losing positions for the first player (Alice) up to n=2024 using the validated representation from Sub-task 2. "
        "Identify the pattern or formula for losing positions to avoid brute force. Verify correctness and optimal play assumptions."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_sc_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before subtask_3: {subtask_desc_3}")
    for i in range(self.max_sc):
        thinking3, answer3 = await cot_sc_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3[i].id}, enumerating losing positions up to 2024, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    best_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[best_answer_3]
    answer3 = answermapping_3[best_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: decompose_simplify_and_sum_components
    debate_instruction_4 = (
        "Sub-task 4: Simplify the enumeration results from Sub-task 3 by expressing losing positions in a closed form or arithmetic progression if a pattern is found. "
        "Compute the total count of such positions for n ≤ 2024. Use debate and reflexion to refine and verify the formula and count."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "Debate"
    }
    print(f"Logging before subtask_4: {subtask_desc_4}")
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying losing positions, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_4[r].append(thinking4)
            all_answer_4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 4: Simplify and count losing positions." , is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing simplified formula and count, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: aggregate_and_combine_values
    cot_instruction_5 = (
        "Sub-task 5: Aggregate and combine the computed values from Sub-task 4 to finalize the answer: the number of positive integers n ≤ 2024 for which Bob has a winning strategy. "
        "Verify the final count through cross-checking or alternative reasoning. Provide the final answer with verification summary."
    )
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before subtask_5: {subtask_desc_5}")
    for i in range(self.max_sc):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, aggregating and verifying final count, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    best_answer_5 = Counter(possible_answers_5).most_common(1)[0][0]
    thinking5 = thinkingmapping_5[best_answer_5]
    answer5 = answermapping_5[best_answer_5]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
