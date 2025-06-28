async def forward_112(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and classify spectral data
    
    # Sub-task 1: Analyze FTIR spectrum data to identify functional groups
    cot_instruction_1 = (
        "Sub-task 1: Analyze the FTIR spectrum data provided (broad absorption at 3000 cm⁻¹, "
        "strong peaks at 1700 and 1650 cm⁻¹) to identify functional groups present in the unknown compound. "
        "Interpret characteristic absorption bands to classify possible functional groups such as hydroxyl, carbonyl, or C=C bonds."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing FTIR spectrum, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Analyze 1H NMR spectrum data focusing on vinyl-hydrogens and other peaks
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the FTIR analysis from Sub-task 1, analyze the 1H NMR spectrum data, "
        "focusing on the presence of vinyl-hydrogens and other observed peaks, to deduce structural features such as types of hydrogens (alkene, alkane, alcohol, etc.) and their environment in the unknown compound."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing 1H NMR spectrum, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most frequent answer for subtask 2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Analyze mass spectrometry data focusing on fragment peak at m/z = 45
    cot_instruction_3 = (
        "Sub-task 3: Analyze the mass spectrometry data, particularly the fragment peak at m/z = 45, "
        "to infer possible fragment structures and molecular weight information that can help narrow down the molecular formula of the unknown compound."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing mass spectrometry data, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Integrate and validate proposed structures

    # Sub-task 4: Integrate FTIR and 1H NMR data to propose possible molecular structures
    cot_reflect_instruction_4 = (
        "Sub-task 4: Integrate the functional group information from FTIR (Sub-task 1) with the structural insights from 1H NMR (Sub-task 2) "
        "to propose possible molecular structures consistent with the observed spectral data."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, integrating FTIR and 1H NMR data, thinking: {thinking4.content}; answer: {answer4.content}")

    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_iter_4 = cot_inputs_4 + [thinking4, answer4]
    for i in range(N_max_4):
        feedback4, correct4 = await critic_agent_4([taskInfo, thinking4, answer4],
                                                  "Please review the proposed molecular structures for consistency with FTIR and 1H NMR data and provide limitations.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs_iter_4.extend([feedback4])
        thinking4, answer4 = await cot_agent_4(cot_inputs_iter_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining molecular structures, thinking: {thinking4.content}; answer: {answer4.content}")
        cot_inputs_iter_4.extend([thinking4, answer4])
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Correlate proposed structures with mass spectrometry fragment data
    debate_instruction_5 = (
        "Sub-task 5: Correlate the proposed molecular structures from Sub-task 4 with the mass spectrometry fragment data (Sub-task 3) "
        "to validate or eliminate possible structures based on fragment patterns and molecular weight consistency."
    )
    debate_roles = ["Proponent", "Skeptic"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4, thinking3, answer3], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4, thinking3, answer3] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, correlating structures with MS data, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1],
                                                    "Sub-task 5: Make final decision on validated molecular structures based on mass spectrometry fragment data.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision on structure validation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Compare validated structures against given chemical formula choices
    cot_instruction_6 = (
        "Sub-task 6: Compare the validated molecular structures and their corresponding chemical formulas against the given choices "
        "(C6H12O, C6H10O, C6H10O2, C6H12O2) to identify the most likely chemical formula of the unknown compound."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, comparing validated structures with formula choices, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
