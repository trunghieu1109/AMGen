async def forward_41(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Extract and understand given quantitative ratios and constants
    cot_instruction_1 = (
        "Sub-task 1: Extract and understand all given quantitative ratios and constants from the problem: "
        "temperature ratios (T1/T2 ≈ 1.4, T2/T3 ≈ 2.3), mass ratios (M1/M2 ≈ 1.15, M2/M3 ≈ 1.35), "
        "and the albedo value (0.3) for all planets. This sets the foundation for further calculations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting quantitative ratios, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 1: Recall and write down relevant physical relationships and formulas
    cot_instruction_2 = (
        "Sub-task 2: Recall and write down the relevant physical relationships and formulas connecting equilibrium temperature, "
        "orbital period, stellar flux, and albedo for planets in circular orbits, especially as used in the TTV context. "
        "Include the formula for equilibrium temperature and Kepler's third law relating orbital period and orbital radius."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, recalling physical formulas, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Derive ratio of orbital radii using equilibrium temperature ratios and albedo
    cot_instruction_3 = (
        "Sub-task 3: Using the equilibrium temperature ratios and the known albedo, derive the ratio of the orbital radii (semi-major axes) between the planets. "
        "Apply the equilibrium temperature formula relating temperature to stellar flux and orbital radius, considering equal albedo."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, deriving orbital radii ratios, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Apply Kepler's third law with mass and orbital radius ratios to find orbital period ratio
    cot_instruction_4 = (
        "Sub-task 4: Using the mass ratios and the orbital radius ratios, apply Kepler's third law to relate the orbital periods of the planets. "
        "Calculate the factor by which the orbital period of Planet3 is larger than that of Planet1."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating orbital period ratio, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2: Compare calculated orbital period ratio with given choices and identify closest match using Debate
    debate_instruction_5 = (
        "Sub-task 5: Compare the calculated orbital period ratio (Planet3/Planet1) with the given multiple-choice options (~4.4, ~33.4, ~10.4, ~3.2) "
        "and identify the closest match."
    )
    debate_roles = ["Proponent", "Skeptic"]
    debate_agents_5 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in debate_roles
    ]
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
            agents.append(f"Debate agent {agent.id}, round {r}, comparing orbital period ratio with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the closest orbital period ratio match.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding closest orbital period ratio, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
