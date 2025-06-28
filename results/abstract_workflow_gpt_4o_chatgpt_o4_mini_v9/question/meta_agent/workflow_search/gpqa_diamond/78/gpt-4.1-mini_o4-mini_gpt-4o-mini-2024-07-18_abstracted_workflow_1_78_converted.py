async def forward_78(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Initial Analysis of Inputs
    # Sub-task 1: Analyze molecular formula C11H12O for structural features
    cot_instruction_1 = (
        "Sub-task 1: Analyze the molecular formula C11H12O of Compound X to determine possible structural features, "
        "including degree of unsaturation and functional groups, providing foundational understanding for spectral analysis."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, molecular formula analysis, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Examine reaction conditions to infer chemical transformation
    cot_instruction_2 = (
        "Sub-task 2: Examine the reaction conditions (reaction with 1,4-diazabicyclo[2.2.2]octane in nonpolar solvent at 403 K for 1 day) "
        "to infer the type of chemical transformation likely undergone by Compound X, informing expected structural changes in the product."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, reaction condition analysis, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Analyze 1H NMR spectral data
    cot_instruction_3 = (
        "Sub-task 3: Analyze the provided 1H NMR spectral data (chemical shifts, multiplicities, integration) of the product "
        "to identify types and environments of hydrogen atoms present, aiding structural elucidation."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, 1H NMR analysis, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Analyze 13C NMR spectral data
    cot_instruction_4 = (
        "Sub-task 4: Analyze the provided 13C NMR spectral data (chemical shifts and number of carbons) of the product "
        "to identify types and environments of carbon atoms present, complementing 1H NMR analysis for structural determination."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, 13C NMR analysis, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 1: Hypothesize product structure integrating molecular formula and reaction insights
    cot_instruction_5 = (
        "Sub-task 5: Integrate molecular formula analysis (Sub-task 1) and reaction condition insights (Sub-task 2) "
        "to hypothesize possible structural framework of the product formed from Compound X after reaction with 1,4-diazabicyclo[2.2.2]octane."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, product structure hypothesis, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 1 continued: Correlate 1H NMR with hypothesized structure
    cot_instruction_6 = (
        "Sub-task 6: Correlate 1H NMR spectral features (Sub-task 3) with hypothesized structural framework (Sub-task 5) "
        "to assign proton environments and validate or refine the proposed product structure."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking3, answer3, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, 1H NMR correlation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Stage 1 continued: Correlate 13C NMR with hypothesized structure
    cot_instruction_7 = (
        "Sub-task 7: Correlate 13C NMR spectral features (Sub-task 4) with hypothesized structural framework (Sub-task 5) "
        "to assign carbon environments and further validate or refine the proposed product structure."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, 13C NMR correlation, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    # Stage 2: Compare integrated NMR-based structural proposal with candidate compounds
    debate_instruction_8 = (
        "Sub-task 8: Compare the integrated NMR-based structural proposal (Sub-tasks 6 and 7) with the four given candidate compounds "
        "(2-styrylepoxide, 2-(4-methylstyryl)oxirane, 2-(1-phenylprop-1-en-2-yl)oxirane, 2-methyl-3-styryloxirane) to identify which candidate best matches the spectral data and reaction context."
    )
    debate_agents_8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]

    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents_8):
            if r == 0:
                thinking8, answer8 = await agent(
                    [taskInfo, thinking6, answer6, thinking7, answer7], debate_instruction_8, r, is_sub_task=True)
            else:
                input_infos_8 = [taskInfo, thinking6, answer6, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1]
                thinking8, answer8 = await agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, candidate comparison, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)

    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8(
        [taskInfo] + all_thinking8[-1] + all_answer8[-1],
        "Sub-task 8: Make final decision on which candidate compound best matches the spectral data and reaction context.",
        is_sub_task=True)
    agents.append(f"Final Decision agent on candidate selection, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    # Stage 3: Select most plausible identity of Compound X
    cot_reflect_instruction_9 = (
        "Sub-task 9: Select the most plausible identity of Compound X by evaluating which candidate compound from Sub-task 8 "
        "conforms best to all spectral data, molecular formula constraints, and reaction conditions, thereby answering the original query."
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_9 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_9 = self.max_round

    cot_inputs_9 = [taskInfo, thinking8, answer8]
    thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_9.id}, selecting most plausible compound, thinking: {thinking9.content}; answer: {answer9.content}")

    for i in range(N_max_9):
        feedback9, correct9 = await critic_agent_9([taskInfo, thinking9, answer9],
                                                  "Please review the selected compound identity and provide limitations.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_9.id}, providing feedback, thinking: {feedback9.content}; answer: {correct9.content}")
        if correct9.content == "True":
            break
        cot_inputs_9.extend([thinking9, answer9, feedback9])
        thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_9.id}, refining compound selection, thinking: {thinking9.content}; answer: {answer9.content}")

    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer
