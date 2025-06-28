async def forward_63(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Analyze chemical species and classify roles
    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the problem statement to identify all chemical species involved: salts A and B, gases formed, "
        "and reagents in tubes №1 (anhydrous Mg(ClO4)2), №2 (Ca(OH)2 solution), №3 (red-hot copper). Classify their roles and properties relevant to the problem."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing chemical species, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    # Stage 0: Determine nature of gases formed
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Based on the analysis from Sub-task 1, determine the nature of gases formed from heating the equimolar mixture of salts A and B at 200°C without air, "
        "considering the reagents used in tubes №1, №2, and №3."
    )
    N_sc_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_2)]
    possible_answers_0_2 = []
    thinkingmapping_0_2 = {}
    answermapping_0_2 = {}
    for i in range(N_sc_0_2):
        thinking_0_2, answer_0_2 = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, determining gases formed, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
        possible_answers_0_2.append(answer_0_2.content)
        thinkingmapping_0_2[answer_0_2.content] = thinking_0_2
        answermapping_0_2[answer_0_2.content] = answer_0_2
    # Choose the most consistent answer (majority vote)
    from collections import Counter
    answer_counts_0_2 = Counter(possible_answers_0_2)
    best_answer_0_2 = answer_counts_0_2.most_common(1)[0][0]
    thinking_0_2 = thinkingmapping_0_2[best_answer_0_2]
    answer_0_2 = answermapping_0_2[best_answer_0_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Assess impact of tubes №1, №2, №3
    # Sub-task 3: Tube №1 (anhydrous Mg(ClO4)2) absorption analysis
    cot_instruction_1_3 = (
        "Sub-task 3: Assess the impact of passing the gas mixture through tube №1 containing anhydrous Mg(ClO4)2, "
        "using the given weight increase of 3.60 g to deduce which gas component(s) were absorbed and their quantities."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, analyzing tube №1 absorption, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    print("Step 1.3: ", sub_tasks[-1])

    # Sub-task 4: Tube №2 (Ca(OH)2 solution) no weight change analysis
    cot_instruction_1_4 = (
        "Sub-task 4: Assess the impact of passing the gas mixture through tube №2 containing Ca(OH)2 solution, "
        "using the information that the weight of this tube did not change, to infer which gas components were not absorbed or reacted."
    )
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await cot_agent_1_4([taskInfo, thinking_1_3, answer_1_3], cot_instruction_1_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_4.id}, analyzing tube №2 no weight change, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    # Sub-task 5: Tube №3 (red-hot copper) reaction analysis
    debate_instruction_1_5 = (
        "Sub-task 5: Assess the impact of passing the gas mixture through tube №3 containing red-hot copper, "
        "using the weight increase of 0.80 g (CuO formed) to deduce which gas component(s) reacted and their quantities."
    )
    debate_agents_1_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_5 = self.max_round
    all_thinking_1_5 = [[] for _ in range(N_max_1_5)]
    all_answer_1_5 = [[] for _ in range(N_max_1_5)]
    for r in range(N_max_1_5):
        for i, agent in enumerate(debate_agents_1_5):
            if r == 0:
                thinking_1_5, answer_1_5 = await agent([taskInfo, thinking_1_4, answer_1_4], debate_instruction_1_5, r, is_sub_task=True)
            else:
                input_infos_1_5 = [taskInfo, thinking_1_4, answer_1_4] + all_thinking_1_5[r-1] + all_answer_1_5[r-1]
                thinking_1_5, answer_1_5 = await agent(input_infos_1_5, debate_instruction_1_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing tube №3 reaction, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
            all_thinking_1_5[r].append(thinking_1_5)
            all_answer_1_5[r].append(answer_1_5)
    final_decision_agent_1_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_5, answer_1_5 = await final_decision_agent_1_5([taskInfo] + all_thinking_1_5[-1] + all_answer_1_5[-1], "Sub-task 5: Make final decision on tube №3 reaction analysis.", is_sub_task=True)
    agents.append(f"Final Decision agent on tube №3 reaction, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_1_5.content}; answer - {answer_1_5.content}")
    print("Step 1.5: ", sub_tasks[-1])

    # Stage 2: Derive gas composition and molar quantities
    cot_reflect_instruction_2_6 = (
        "Sub-task 6: Derive the composition and molar quantities of the gases formed from the initial salts A and B, "
        "using the data from weight changes in tubes №1 and №3, unchanged weight in tube №2, and volume of remaining gas C (2.24 L at STP)."
    )
    cot_agent_2_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_6 = self.max_round
    cot_inputs_2_6 = [taskInfo, thinking_1_5, answer_1_5]
    thinking_2_6, answer_2_6 = await cot_agent_2_6(cot_inputs_2_6, cot_reflect_instruction_2_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_6.id}, deriving gas composition, thinking: {thinking_2_6.content}; answer: {answer_2_6.content}")
    for i in range(N_max_2_6):
        feedback_2_6, correct_2_6 = await critic_agent_2_6([taskInfo, thinking_2_6, answer_2_6], "Critically evaluate the gas composition derivation and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_6.id}, feedback on gas composition, thinking: {feedback_2_6.content}; answer: {correct_2_6.content}")
        if correct_2_6.content == "True":
            break
        cot_inputs_2_6.extend([thinking_2_6, answer_2_6, feedback_2_6])
        thinking_2_6, answer_2_6 = await cot_agent_2_6(cot_inputs_2_6, cot_reflect_instruction_2_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_6.id}, refining gas composition, thinking: {thinking_2_6.content}; answer: {answer_2_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_2_6.content}; answer - {answer_2_6.content}")
    print("Step 2.6: ", sub_tasks[-1])

    # Sub-task 7: Determine molecular formulas of salts A and B
    cot_reflect_instruction_2_7 = (
        "Sub-task 7: Determine the molecular formulas of salts A and B based on the gas composition and quantities derived, "
        "considering the equimolar mixture and total initial mass of 7.20 g."
    )
    cot_agent_2_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_7 = self.max_round
    cot_inputs_2_7 = [taskInfo, thinking_2_6, answer_2_6]
    thinking_2_7, answer_2_7 = await cot_agent_2_7(cot_inputs_2_7, cot_reflect_instruction_2_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_7.id}, determining molecular formulas, thinking: {thinking_2_7.content}; answer: {answer_2_7.content}")
    for i in range(N_max_2_7):
        feedback_2_7, correct_2_7 = await critic_agent_2_7([taskInfo, thinking_2_7, answer_2_7], "Critically evaluate the molecular formulas determination and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_7.id}, feedback on molecular formulas, thinking: {feedback_2_7.content}; answer: {correct_2_7.content}")
        if correct_2_7.content == "True":
            break
        cot_inputs_2_7.extend([thinking_2_7, answer_2_7, feedback_2_7])
        thinking_2_7, answer_2_7 = await cot_agent_2_7(cot_inputs_2_7, cot_reflect_instruction_2_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_7.id}, refining molecular formulas, thinking: {thinking_2_7.content}; answer: {answer_2_7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking_2_7.content}; answer - {answer_2_7.content}")
    print("Step 2.7: ", sub_tasks[-1])

    # Stage 3: Calculate total number of atoms in salts A and B
    cot_reflect_instruction_3_8 = (
        "Sub-task 8: Calculate the total number of all atoms in the combined formulas of salts A and B, "
        "using the molecular formulas determined and stoichiometric relationships from previous subtasks."
    )
    cot_agent_3_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_8 = self.max_round
    cot_inputs_3_8 = [taskInfo, thinking_2_7, answer_2_7]
    thinking_3_8, answer_3_8 = await cot_agent_3_8(cot_inputs_3_8, cot_reflect_instruction_3_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_8.id}, calculating total atoms, thinking: {thinking_3_8.content}; answer: {answer_3_8.content}")
    for i in range(N_max_3_8):
        feedback_3_8, correct_3_8 = await critic_agent_3_8([taskInfo, thinking_3_8, answer_3_8], "Critically evaluate the total atoms calculation and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_8.id}, feedback on total atoms, thinking: {feedback_3_8.content}; answer: {correct_3_8.content}")
        if correct_3_8.content == "True":
            break
        cot_inputs_3_8.extend([thinking_3_8, answer_3_8, feedback_3_8])
        thinking_3_8, answer_3_8 = await cot_agent_3_8(cot_inputs_3_8, cot_reflect_instruction_3_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_8.id}, refining total atoms, thinking: {thinking_3_8.content}; answer: {answer_3_8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking_3_8.content}; answer - {answer_3_8.content}")
    print("Step 3.8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_8, answer_3_8, sub_tasks, agents)
    return final_answer
