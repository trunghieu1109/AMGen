async def forward_4(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Extract and define operator components and direction vector n
    cot_instruction_0_1 = (
        "Sub-task 1: Extract and clearly define the quantum mechanical operator components Px, Py, and Pz "
        "as 2x2 matrices multiplied by ħ/2, including their explicit matrix elements, and identify the direction vector n in the x-z plane."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, extracting operator components and direction vector, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Formulate the operator P along the arbitrary direction n in the x-z plane as a linear combination of Px and Pz "
        "(since Py component is zero for n in x-z plane), using the extracted matrices from subtask_1."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, formulating operator P along n, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Set up eigenvalue equation and solve for eigenvector
    cot_instruction_1_3 = (
        "Sub-task 3: Set up the eigenvalue equation P|\u03C8> = (+\u210F/2)|\u03C8> using the combined operator matrix from subtask_2 and eigenvalue +\u210F/2."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, setting up eigenvalue equation, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    print("Step 1.3: ", sub_tasks[-1])

    cot_instruction_1_4 = (
        "Sub-task 4: Solve the eigenvalue equation from subtask_3 to find the eigenvector |\u03C8> corresponding to eigenvalue +\u210F/2, "
        "expressing the eigenvector components in terms of \u03B8 (the angle defining direction n in the x-z plane)."
    )
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await cot_agent_1_4([taskInfo, thinking_1_3, answer_1_3], cot_instruction_1_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_4.id}, solving eigenvalue equation, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    cot_instruction_1_5 = (
        "Sub-task 5: Normalize the eigenvector obtained in subtask_4 to ensure it has unit norm, yielding the normalized eigenvector components."
    )
    cot_agent_1_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_5, answer_1_5 = await cot_agent_1_5([taskInfo, thinking_1_4, answer_1_4], cot_instruction_1_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_5.id}, normalizing eigenvector, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_1_5.content}; answer - {answer_1_5.content}")
    print("Step 1.5: ", sub_tasks[-1])

    # Stage 2: Compare normalized eigenvector with given choices to identify correct form
    debate_instruction_2_6 = (
        "Sub-task 6: Compare the normalized eigenvector components from subtask_5 with the given multiple-choice options "
        "to identify the correct form of the eigenvector."
    )
    debate_agents_2_6 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_2_6 = self.max_round
    all_thinking_2_6 = [[] for _ in range(N_max_2_6)]
    all_answer_2_6 = [[] for _ in range(N_max_2_6)]

    for r in range(N_max_2_6):
        for i, agent in enumerate(debate_agents_2_6):
            input_infos_2_6 = [taskInfo, thinking_1_5, answer_1_5]
            if r > 0:
                input_infos_2_6.extend(all_thinking_2_6[r-1])
                input_infos_2_6.extend(all_answer_2_6[r-1])
            thinking_2_6, answer_2_6 = await agent(input_infos_2_6, debate_instruction_2_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing normalized eigenvector with choices, thinking: {thinking_2_6.content}; answer: {answer_2_6.content}")
            all_thinking_2_6[r].append(thinking_2_6)
            all_answer_2_6[r].append(answer_2_6)

    final_decision_agent_2_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_6, answer_2_6 = await final_decision_agent_2_6(
        [taskInfo] + all_thinking_2_6[-1] + all_answer_2_6[-1],
        "Sub-task 6: Make final decision on the correct normalized eigenvector form.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent on identifying eigenvector, thinking: {thinking_2_6.content}; answer: {answer_2_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_2_6.content}; answer - {answer_2_6.content}")
    print("Step 2.6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_6, answer_2_6, sub_tasks, agents)
    return final_answer
