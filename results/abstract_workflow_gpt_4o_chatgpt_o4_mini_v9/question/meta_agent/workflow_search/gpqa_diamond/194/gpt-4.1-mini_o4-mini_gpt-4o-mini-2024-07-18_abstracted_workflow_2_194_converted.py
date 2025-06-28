async def forward_194(self, taskInfo):
    from collections import Counter
    import math

    print("Task Requirement: ", taskInfo)

    sub_tasks = []
    agents = []

    # Stage 1: Analyze and classify given planetary system parameters
    cot_instruction_1 = (
        "Sub-task 1: Analyze and classify the given planetary system parameters: radius of the first planet (1 Earth radius), "
        "radius of the star (1.5 solar radii), orbital period of the first planet (3 days), and transit impact parameter (0.2). "
        "Understand their physical meanings and relationships relevant to transit and occultation geometry."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing planetary parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 1: Sub-task 2: Determine geometric constraints for transit and occultation
    cot_instruction_2 = (
        "Sub-task 2: Determine the geometric constraints for a planet to exhibit both transit and occultation events given the star radius, orbital radius, "
        "and impact parameter of the first planet. Understand inclination limits and how impact parameter relates to transit geometry."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, determining geometric constraints, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 1: Sub-task 3: Calculate orbital radius (semi-major axis) of first planet
    cot_instruction_3 = (
        "Sub-task 3: Calculate the orbital radius (semi-major axis) of the first planet using its orbital period (3 days) and stellar parameters, "
        "assuming circular orbit and solar mass inferred from star radius."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating orbital radius, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Sub-task 4: Derive max allowed orbital inclination range for first planet
    cot_instruction_4 = (
        "Sub-task 4: Using the orbital radius of the first planet and the transit impact parameter, derive the maximum allowed orbital inclination range "
        "for the first planet to produce the observed transit impact parameter of 0.2."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, deriving inclination range, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2: Sub-task 5: Apply geometric constraints to second planet to find max orbital radius
    cot_instruction_5 = (
        "Sub-task 5: Apply the geometric constraints for transit and occultation to the second planet (2.5 Earth radii) sharing the same orbital plane, "
        "to find the maximum orbital radius at which both transit and occultation can still occur."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, applying constraints to second planet, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Stage 2: Sub-task 6: Convert max orbital radius of second planet to max orbital period
    cot_instruction_6 = (
        "Sub-task 6: Convert the maximum orbital radius of the second planet into the corresponding maximum orbital period using Kepler's third law and stellar parameters."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5, thinking3, answer3], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, converting radius to period, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Stage 2: Sub-task 7: Compare calculated max orbital period with choices and select closest
    debate_instruction_7 = (
        "Sub-task 7: Compare the calculated maximum orbital period of the second planet with the provided multiple-choice options (~7.5, ~33.5, ~37.5, ~12.5 days) "
        "and select the closest matching value."
    )
    debate_roles = ["Option Advocate 1", "Option Advocate 2"]
    debate_agents_7 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in debate_roles
    ]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]

    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent(
                    [taskInfo, thinking6, answer6], debate_instruction_7, r, is_sub_task=True
                )
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r - 1] + all_answer7[r - 1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing max orbital period with choices, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)

    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7(
        [taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the closest matching maximum orbital period.", is_sub_task=True
    )
    agents.append(f"Final Decision agent, selecting closest max orbital period, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
