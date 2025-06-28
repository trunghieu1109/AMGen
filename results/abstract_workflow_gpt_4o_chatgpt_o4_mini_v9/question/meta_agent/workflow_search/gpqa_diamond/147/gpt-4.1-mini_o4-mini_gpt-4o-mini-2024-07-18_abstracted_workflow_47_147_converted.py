async def forward_147(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Identify element Y and the five binary fluorine compounds (A1 to A5)
    # Sub-task 1: Identify element Y and compounds A1 to A5 based on given properties
    cot_instruction_1 = (
        "Sub-task 1: Identify element Y and the five binary fluorine compounds (A1 to A5) "
        "based on the given chemical and physical properties, including decomposition behavior, fluorine content, and reactivity with xenon. "
        "Analyze the clues: A1 decomposes at 293 K into A2 and fluorine, A1 oxidizes xenon, A3 is obtained by fluorination of Y, "
        "and A5 is formed by reaction of A4 with Y and decomposes in water into two substances."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying element Y and compounds A1 to A5, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Determine chemical formulas and molecular weights of A1, A2, A3
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1 output, determine the chemical formulas and molecular weights of compounds A1, A2, and A3. "
        "Use the data: A1 decomposes at 293 K into A2 and fluorine, A2 has fluorine content 31.96%, and A3 is obtained by direct fluorination of element Y. "
        "Consider all possible molecular formulas consistent with these facts."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining formulas and weights of A1, A2, A3, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer (majority vote or best reasoning)
    from collections import Counter
    answer_counts_2 = Counter(possible_answers_2)
    best_answer_2 = answer_counts_2.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Analyze reaction of A4 with Y and decomposition of A5 in water
    cot_reflect_instruction_3 = (
        "Sub-task 3: Based on Sub-task 1 output, analyze the reaction of A4 with element Y in 1:1 molar ratio to form A5, "
        "and the decomposition of A5 in water into two substances. Infer the nature and formula of A4 and A5. "
        "Filter valid scenarios consistent with previous findings."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, analyzing A4 and A5 reactions, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                               "Review the analysis of A4 and A5 reactions and decomposition, provide limitations.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback on A4/A5 analysis, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining A4/A5 analysis, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Calculate molecular weight range of A4
    # Sub-task 4: Calculate or estimate molecular weight range of A4 based on formulas and stoichiometry
    cot_instruction_4 = (
        "Sub-task 4: Using the chemical formulas and molecular weights of A1, A2, A3, and A5, and the stoichiometry of their reactions, "
        "calculate or estimate the molecular weight range of compound A4."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating molecular weight range of A4, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Compare calculated molecular weight range of A4 with given options and select correct range
    debate_instruction_5 = (
        "Sub-task 5: Based on the calculated molecular weight range of A4, compare it with the given multiple-choice options: "
        "220-240, 140-160, 160-180, 110-130, and select the correct range. "
        "Debate the reasoning and finalize the choice."
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
            agents.append(f"Debate agent {agent.id}, round {r}, debating molecular weight range of A4, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on molecular weight range of A4.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding molecular weight range of A4, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
