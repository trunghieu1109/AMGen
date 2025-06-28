async def forward_50(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze physical setup and classify problem type
    # Sub-task 1: Analyze physical setup
    cot_instruction_1 = (
        "Sub-task 1: Analyze the physical setup where a charge q is placed at distance d from the center of a grounded conducting sphere of radius R. "
        "Identify relevant physical principles and boundary conditions, especially that the sphere is grounded (potential on sphere surface is zero)."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing physical setup, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    # Sub-task 2: Classify problem type as electrostatics with image charges
    cot_instruction_2 = (
        "Sub-task 2: Based on the analysis from Sub-task 1, classify the problem as an electrostatics problem involving image charges to satisfy the boundary condition of zero potential on the grounded conducting sphere surface."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, classifying problem type, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most common answer
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinkingmapping_2[most_common_answer_2].content}; answer - {most_common_answer_2}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Derive image charge configuration and potential energy expression
    # Sub-task 3: Derive image charge configuration
    cot_instruction_3 = (
        "Sub-task 3: Derive the image charge configuration that replaces the grounded conducting sphere with an equivalent system of charges to satisfy the zero potential boundary condition on the sphere surface. "
        "Use the method of image charges to find magnitude and position of the image charge."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinkingmapping_2[most_common_answer_2], answermapping_2[most_common_answer_2]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, deriving image charge configuration, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 1.3: ", sub_tasks[-1])

    # Sub-task 4: Express potential energy in terms of real and image charges
    cot_instruction_4 = (
        "Sub-task 4: Express the potential energy of the system in terms of the real charge q and the image charge, using their magnitudes and positions derived from the image charge method."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, expressing potential energy, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Calculate net potential energy quantitatively
    # Sub-task 5: Calculate net potential energy
    debate_instruction_5 = (
        "Sub-task 5: Based on the expression from Sub-task 4, calculate the net potential energy of the system by evaluating the interaction energy between the real charge and the image charge, "
        "considering the distance between them and Coulomb constant k."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating net potential energy, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on net potential energy expression.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating net potential energy, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 2.5: ", sub_tasks[-1])

    # Stage 3: Compare calculated expression with given choices
    cot_reflect_instruction_6 = (
        "Sub-task 6: Compare the calculated expression for net potential energy with the given choices to identify the correct formula for U. "
        "Justify the selection based on the derived formula and constants."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_reflect_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, comparing with choices, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 3.6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
