async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Identify known parameters and recall formulas

    # Sub-task 1: Identify and list all known parameters for Planet_1 and Planet_2 and their host stars
    cot_instruction_1 = (
        "Sub-task 1: Identify and list all known parameters for Planet_1 and Planet_2 and their host stars from the query, "
        "including minimum masses, orbital periods, orbit eccentricities, host star masses, and host star radii. "
        "This sets the foundation for further calculations by clearly defining the input variables."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying known parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Recall and write down the formula for geometric transit probability
    cot_instruction_2 = (
        "Sub-task 2: Recall and write down the formula for the geometric transit probability of a planet, "
        "which depends on the ratio of the host star radius to the orbital semi-major axis, assuming circular orbits. "
        "This formula is essential to compute the transit probabilities for both planets."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, recalling transit probability formula, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Derive relationship between orbital period and semi-major axis using Kepler's third law
    cot_instruction_3 = (
        "Sub-task 3: Derive the relationship between orbital period and semi-major axis using Kepler's third law, "
        "incorporating the host star mass. This will allow calculation of the semi-major axis for each planet from their orbital periods and host star masses."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, deriving Kepler's law relation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Calculate semi-major axes and transit probabilities, then compare

    # Sub-task 4: Calculate semi-major axis for Planet_1
    cot_instruction_4 = (
        "Sub-task 4: Calculate the semi-major axis for Planet_1 using its orbital period and the mass of its host star, "
        "applying the relationship from Kepler's third law. This step uses the known orbital period (three times shorter than Planet_2) "
        "and the host star mass (twice that of Planet_2's host star)."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating semi-major axis Planet_1, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Calculate semi-major axis for Planet_2
    cot_instruction_5 = (
        "Sub-task 5: Calculate the semi-major axis for Planet_2 using its orbital period and the mass of its host star, "
        "applying the relationship from Kepler's third law. This uses the known orbital period and host star mass for Planet_2."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking1, answer1, thinking3, answer3], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating semi-major axis Planet_2, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Compute transit probability for Planet_1
    cot_instruction_6 = (
        "Sub-task 6: Compute the transit probability for Planet_1 using the host star radius and the semi-major axis calculated in Sub-task 4, "
        "applying the transit probability formula for circular orbits."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking1, answer1, thinking2, answer2, thinking4, answer4], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, computing transit probability Planet_1, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Sub-task 7: Compute transit probability for Planet_2
    cot_instruction_7 = (
        "Sub-task 7: Compute the transit probability for Planet_2 using the host star radius and the semi-major axis calculated in Sub-task 5, "
        "applying the transit probability formula for circular orbits."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking1, answer1, thinking2, answer2, thinking5, answer5], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, computing transit probability Planet_2, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    # Sub-task 8: Compare transit probabilities by calculating their ratio
    cot_instruction_8 = (
        "Sub-task 8: Compare the transit probabilities of Planet_1 and Planet_2 by calculating their ratio to determine which planet has the higher probability of transiting. "
        "This comparison will directly inform the researchers' choice."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking6, answer6, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, comparing transit probabilities, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Subtask 8 answer: ", sub_tasks[-1])

    # Sub-task 9: Match calculated ratio to closest given choice and identify preferred planet
    cot_instruction_9 = (
        "Sub-task 9: Match the calculated ratio of transit probabilities to the closest given choice in the query options, "
        "and identify which planet is preferred based on the highest transit probability and the approximate ratio value."
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking8, answer8], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, matching ratio to choice and identifying preferred planet, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Subtask 9 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer
