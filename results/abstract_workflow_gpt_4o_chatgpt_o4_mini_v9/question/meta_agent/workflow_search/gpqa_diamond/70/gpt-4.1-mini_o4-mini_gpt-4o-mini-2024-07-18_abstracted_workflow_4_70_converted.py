async def forward_70(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Extract orbital period ratios and clarify albedo assumption

    # Sub-task 1: Identify and extract the given orbital period ratios
    cot_instruction_1 = (
        "Sub-task 1: Extract the orbital period ratios of the five exoplanets (Planet_1 through Planet_5) "
        "from the query, noting the ratio is 1:2:2.5:3.5:5 and that the orbits are circular and in resonance."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracted orbital period ratios, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Clarify assumption that all planets have the same albedo and its implication
    cot_instruction_2 = (
        "Sub-task 2: Clarify the assumption that all planets have the same albedo and explain its implication "
        "on equilibrium temperature calculations, especially the inverse square root relation with orbital radius."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, clarified albedo assumption, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 1: Compute relative orbital radii of Planet_2 and Planet_4 using Kepler's third law

    cot_instruction_3 = (
        "Sub-task 3: Using the orbital period ratios from Sub-task 1, compute the relative orbital radii of Planet_2 and Planet_4 "
        "by applying Kepler's third law (orbital radius proportional to the cube root of the square of the orbital period), "
        "assuming the central star's mass is constant."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, computed relative orbital radii, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Derive formula for equilibrium temperature and calculate temperature ratio

    # Sub-task 4: Derive formula for equilibrium temperature assuming circular orbit and same albedo
    cot_instruction_4 = (
        "Sub-task 4: Derive the formula for equilibrium temperature of a planet assuming circular orbit and same albedo, "
        "which relates equilibrium temperature inversely to the square root of the orbital radius."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, derived equilibrium temperature formula, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Calculate the ratio of equilibrium temperatures between Planet_4 and Planet_2
    cot_instruction_5 = (
        "Sub-task 5: Calculate the ratio of equilibrium temperatures between Planet_4 and Planet_2 by combining the orbital radius ratio "
        "from Sub-task 3 and the temperature-radius relationship from Sub-task 4."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculated equilibrium temperature ratio, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Stage 3: Compare calculated temperature ratio with provided options and select closest

    debate_instruction_6 = (
        "Sub-task 6: Compare the calculated equilibrium temperature ratio from Sub-task 5 with the provided multiple-choice options (~0.75, ~0.83, ~0.69, ~0.57) "
        "and select the closest matching value."
    )
    debate_agents_6 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]

    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting closest temperature ratio, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the closest equilibrium temperature ratio.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting closest temperature ratio, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
