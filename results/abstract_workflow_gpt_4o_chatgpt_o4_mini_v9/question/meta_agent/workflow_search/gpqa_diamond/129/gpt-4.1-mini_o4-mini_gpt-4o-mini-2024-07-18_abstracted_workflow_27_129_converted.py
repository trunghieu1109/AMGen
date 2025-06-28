async def forward_129(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Identify wavelength range, understand absorption and redshift relation

    # Sub-task 1: Identify relevant wavelength range for optical ground-based telescopes (visible light range)
    cot_instruction_1 = (
        "Sub-task 1: Identify the typical wavelength range of visible light detectable by optical ground-based telescopes, "
        "considering atmospheric transmission and typical optical bands."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying visible wavelength range, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Understand Lyman alpha absorption and redshift effect on observed wavelength
    cot_instruction_2 = (
        "Sub-task 2: Explain how large clouds of neutral hydrogen absorb light at Lyman alpha wavelength (~1216 Angstrom) "
        "and how redshift z affects the observed wavelength of this absorption feature."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, explaining Lyman alpha absorption and redshift, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Define relationship observed_wavelength = emitted_wavelength * (1 + z)
    cot_reflect_instruction_3 = (
        "Sub-task 3: Based on Sub-task 2, define the formula relating observed wavelength, emitted wavelength (1216 Angstrom), and redshift z. "
        "Explain the formula and its implications."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, defining redshift formula, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                               "Critically evaluate the redshift formula explanation for correctness and completeness.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback on redshift formula, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining redshift formula explanation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Calculate minimum redshift and compare with choices

    # Sub-task 4: Calculate minimum redshift z such that Lyman alpha line is redshifted into optical range
    cot_instruction_4 = (
        "Sub-task 4: Using the visible wavelength range from Sub-task 1 and the redshift formula from Sub-task 3, "
        "calculate the minimum redshift z so that the Lyman alpha absorption line (~1216 Angstrom) is observed within the optical wavelength range detectable by ground-based telescopes."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating minimum redshift, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Compare calculated minimum redshift with given multiple-choice options and select closest lower limit
    debate_instruction_5 = (
        "Sub-task 5: Based on the calculated minimum redshift from Sub-task 4, compare it with the given multiple-choice options: 1.2, 1.9, 2.4, 3. "
        "Select the closest lower limit redshift value for detection with optical ground-based telescopes."
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
            agents.append(f"Debate agent {agent.id}, round {r}, comparing redshift with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1],
                                                    "Sub-task 5: Make final decision on the closest lower limit redshift value.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, selecting closest lower limit redshift, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
