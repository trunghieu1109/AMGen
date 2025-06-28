async def forward_95(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 0: Analyze and interpret inputs
    # Sub-task 1: Convert angular size from degrees to radians and interpret inputs
    cot_instruction_1 = (
        "Sub-task 1: Analyze and convert the given angular size θ = 10^-17 degrees to radians, "
        "and interpret the meaning of angular size and distance d = 10^10 parsecs in the context of the black hole's event horizon apparent size."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, converting angular size and interpreting inputs, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # Sub-task 2: Identify physical meaning of angular size and distance to estimate Schwarzschild radius
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the conversion and interpretation from Sub-task 1, identify the physical meaning of the angular size and distance, "
        "and explain how to estimate the physical radius (Schwarzschild radius) of the black hole using these values."
    )
    N_sc = self.max_sc if hasattr(self, 'max_sc') else 3
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identifying physical meaning and estimation method, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer (majority vote)
    counter_2 = Counter(possible_answers_2)
    best_answer_2 = counter_2.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # Stage 1: Calculate Schwarzschild radius, mass, and entropy
    # Sub-task 3: Calculate Schwarzschild radius using small angle approximation
    cot_instruction_3 = (
        "Sub-task 3: Calculate the physical Schwarzschild radius of the black hole using the angular size (in radians) "
        "and distance (converted to meters) applying the small angle approximation: radius = angular size × distance."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating Schwarzschild radius, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    # Sub-task 4: Compute mass from Schwarzschild radius using formula r_s = 2GM/c^2 rearranged to M = r_s c^2 / (2G)
    cot_instruction_4 = (
        "Sub-task 4: Compute the mass of the black hole from the Schwarzschild radius using the formula r_s = 2GM/c^2, "
        "rearranged to solve for mass M = r_s c^2 / (2G)."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, computing mass from Schwarzschild radius, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    # Sub-task 5: Calculate entropy using Bekenstein-Hawking formula S = k * A / (4 * l_p^2), where A = 4πr_s^2
    cot_instruction_5 = (
        "Sub-task 5: Calculate the entropy of the black hole using the Bekenstein-Hawking entropy formula: "
        "S = k * A / (4 * l_p^2), where A = 4πr_s^2 is the event horizon area."
    )
    debate_roles_5 = ["Pro", "Con"] if hasattr(self, 'debate_role') else ["Pro", "Con"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles_5]
    N_max_5 = self.max_round if hasattr(self, 'max_round') else 2
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], cot_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, cot_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating entropy, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the entropy calculation.", is_sub_task=True)
    agents.append(f"Final Decision agent on entropy calculation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])
    
    # Stage 2: Estimate order of magnitude and select closest choice
    # Sub-task 6: Estimate order of magnitude of entropy and compare with given choices
    cot_instruction_6 = (
        "Sub-task 6: Estimate the order of magnitude of the calculated entropy value from Sub-task 5, "
        "and compare it with the given choices (10^59, 10^62, 10^65, 10^66 J/K) to select the closest match."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, estimating order of magnitude and selecting closest choice, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
