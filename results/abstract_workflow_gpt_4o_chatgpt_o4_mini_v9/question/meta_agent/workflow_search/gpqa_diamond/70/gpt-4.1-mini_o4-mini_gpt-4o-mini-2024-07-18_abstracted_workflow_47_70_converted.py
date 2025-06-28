async def forward_70(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Extract orbital periods and restate assumptions

    # Sub-task 1: Identify and extract orbital periods of the five exoplanets
    cot_instruction_1 = (
        "Sub-task 1: Extract the orbital periods of Planet_1 through Planet_5 from the given ratio 1:2:2.5:3.5:5, "
        "expressing them in consistent relative units for further calculations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting orbital periods, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Restate assumption about albedo and circular orbits
    cot_instruction_2 = (
        "Sub-task 2: Restate the assumption that all planets have the same albedo and circular orbits, "
        "implying equilibrium temperature depends primarily on their distance from the star, "
        "which relates to orbital periods via Kepler's third law."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, restating assumptions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Calculate orbital radii and temperature ratio

    # Sub-task 3: Calculate relative orbital radii of Planet_2 and Planet_4 using Kepler's third law
    cot_instruction_3 = (
        "Sub-task 3: Using the orbital period ratios and assumption of circular orbits, calculate the relative orbital radii "
        "of Planet_2 and Planet_4 by applying Kepler's third law (orbital radius proportional to the cube root of the square of the orbital period)."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating orbital radii, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Derive formula for equilibrium temperature and calculate temperature ratio Planet_4 / Planet_2
    cot_instruction_4 = (
        "Sub-task 4: Derive the formula for equilibrium temperature assuming equal albedo and circular orbits, "
        "showing temperature is inversely proportional to the square root of orbital radius, then calculate the ratio "
        "of equilibrium temperatures between Planet_4 and Planet_2 using the orbital radii from Sub-task 3."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, deriving temperature formula and calculating ratio, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Compare calculated temperature ratio with provided choices and identify closest match
    debate_instruction_5 = (
        "Sub-task 5: Compare the calculated temperature ratio from Sub-task 4 with the provided multiple-choice options (~0.75, ~0.83, ~0.69, ~0.57) "
        "and identify the closest matching choice."
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
            agents.append(f"Debate agent {agent.id}, round {r}, comparing temperature ratio and choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the closest matching temperature ratio choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding closest temperature ratio choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
