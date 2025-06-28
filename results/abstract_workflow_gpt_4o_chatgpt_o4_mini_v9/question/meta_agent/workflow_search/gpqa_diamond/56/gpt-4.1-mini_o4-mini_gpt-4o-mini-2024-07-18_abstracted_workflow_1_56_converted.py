async def forward_56(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Identify and classify phase shifts and physical context
    cot_instruction_0_1 = (
        "Sub-task 1: Identify and classify the given phase shifts (δ0=90°, δ1=67°, δ2=55°, δ3=30°, δ4=13°) "
        "and the physical context (elastic scattering of 50 MeV electrons from a nuclear target). Confirm that only these phase shifts are relevant and others can be ignored as per the problem statement."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, identifying and classifying phase shifts, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Analyze the physical meaning of the phase shifts in the context of partial wave expansion of the scattering amplitude, "
        "and identify the formula relating phase shifts to the scattering amplitude, specifically the imaginary part along the incident beam direction."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, analyzing physical meaning and formula, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Derive expression for scattering amplitude and convert phase shifts
    cot_instruction_1_3 = (
        "Sub-task 3: Derive the expression for the scattering amplitude along the incident beam direction (θ=0) "
        "using the partial wave expansion and the given phase shifts, focusing on the imaginary part of the amplitude."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, deriving scattering amplitude expression, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    print("Step 1.3: ", sub_tasks[-1])

    cot_instruction_1_4 = (
        "Sub-task 4: Convert the given phase shifts from degrees to radians as required for the calculation of the scattering amplitude."
    )
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await cot_agent_1_4([taskInfo, thinking_0_1, answer_0_1], cot_instruction_1_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_4.id}, converting phase shifts to radians, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Calculate imaginary part and ensure units
    cot_instruction_2_5 = (
        "Sub-task 5: Calculate the imaginary part of the scattering amplitude by summing over the partial waves "
        "using the converted phase shifts and the derived formula from subtask 3, considering the angular momentum quantum numbers l=0 to 4."
    )
    cot_agent_2_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_2_5, answer_2_5 = await cot_agent_2_5([taskInfo, thinking_1_3, answer_1_3, thinking_1_4, answer_1_4], cot_instruction_2_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_5.id}, calculating imaginary part of scattering amplitude, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_2_5.content}; answer - {answer_2_5.content}")
    print("Step 2.5: ", sub_tasks[-1])

    cot_instruction_2_6 = (
        "Sub-task 6: Ensure the units of the calculated imaginary part of the scattering amplitude are consistent with the given answer choices (in femtometers, fm). "
        "Apply any necessary unit conversions."
    )
    cot_agent_2_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_2_6, answer_2_6 = await cot_agent_2_6([taskInfo, thinking_2_5, answer_2_5], cot_instruction_2_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_6.id}, ensuring unit consistency, thinking: {thinking_2_6.content}; answer: {answer_2_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_2_6.content}; answer - {answer_2_6.content}")
    print("Step 2.6: ", sub_tasks[-1])

    # Stage 3: Compare with answer choices and select closest
    debate_instruction_3_7 = (
        "Sub-task 7: Compare the calculated imaginary part of the scattering amplitude with the provided answer choices "
        "(177.675 fm, 87163.4 fm, 251.271 fm, 355.351 fm) and select the closest matching value."
    )
    debate_agents_3_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_7 = self.max_round
    all_thinking_3_7 = [[] for _ in range(N_max_3_7)]
    all_answer_3_7 = [[] for _ in range(N_max_3_7)]

    for r in range(N_max_3_7):
        for i, agent in enumerate(debate_agents_3_7):
            if r == 0:
                thinking_3_7, answer_3_7 = await agent([taskInfo, thinking_2_6, answer_2_6], debate_instruction_3_7, r, is_sub_task=True)
            else:
                input_infos_3_7 = [taskInfo, thinking_2_6, answer_2_6] + all_thinking_3_7[r-1] + all_answer_3_7[r-1]
                thinking_3_7, answer_3_7 = await agent(input_infos_3_7, debate_instruction_3_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing with answer choices, thinking: {thinking_3_7.content}; answer: {answer_3_7.content}")
            all_thinking_3_7[r].append(thinking_3_7)
            all_answer_3_7[r].append(answer_3_7)

    final_decision_agent_3_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_7, answer_3_7 = await final_decision_agent_3_7([taskInfo] + all_thinking_3_7[-1] + all_answer_3_7[-1], "Sub-task 7: Make final decision on the closest matching imaginary part of scattering amplitude.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting closest answer, thinking: {thinking_3_7.content}; answer: {answer_3_7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking_3_7.content}; answer - {answer_3_7.content}")
    print("Step 3.7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_7, answer_3_7, sub_tasks, agents)
    return final_answer
