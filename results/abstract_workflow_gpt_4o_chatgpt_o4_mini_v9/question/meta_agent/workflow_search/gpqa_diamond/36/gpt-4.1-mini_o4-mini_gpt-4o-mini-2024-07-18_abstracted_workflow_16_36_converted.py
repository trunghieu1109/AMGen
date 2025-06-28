async def forward_36(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 1: Sequential Chemical Structure Determination
    
    # Sub-task 1: Identify and draw the chemical structure of compound A
    cot_instruction_1 = (
        "Sub-task 1: Identify and draw the chemical structure of compound A formed from the reaction of propionaldehyde with EDT and BF3, "
        "as this is the starting point for subsequent transformations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, determining structure of compound A, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    
    # Sub-task 2: Determine structure of compound B by reaction of A with BuLi
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the structure of compound A, determine the chemical structure of compound B formed by reaction with BuLi. "
        "Consider possible reaction pathways and intermediates."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining structure of compound B, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Select the most consistent answer (majority vote)
    from collections import Counter
    answer_counts_2 = Counter(possible_answers_2)
    best_answer_2 = answer_counts_2.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    
    # Sub-task 3: Determine structure of compound C by reaction of B with bromoethane
    cot_instruction_3 = (
        "Sub-task 3: Using the structure of compound B, determine the chemical structure of compound C formed by reaction with bromoethane. "
        "Use Chain-of-Thought reasoning to consider possible alkylation sites and outcomes."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, determining structure of compound C, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    
    # Sub-task 4: Determine structure of compound D by reaction of C with HgCl2, H2O, and H+
    cot_instruction_4 = (
        "Sub-task 4: Based on the structure of compound C, determine the chemical structure of compound D formed by reaction with HgCl2, H2O, and H+. "
        "Consider hydration and organomercury intermediate transformations."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, determining structure of compound D, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    
    # Sub-task 5: Determine structure of final product E by reaction of D with PPh3, 3-bromopentane, and BuLi
    debate_instruction_5 = (
        "Sub-task 5: Based on the structure of compound D, determine the chemical structure of the final product E formed by reaction with PPh3, 3-bromopentane, and BuLi. "
        "Use debate among agents to consider possible reaction mechanisms and final structures."
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
            agents.append(f"Debate agent {agent.id}, round {r}, determining structure of compound E, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the chemical structure of compound E.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining structure of compound E, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    
    # Stage 2: Analyze final product E for unique 13C-NMR signals
    
    # Sub-task 6: Analyze the final product E's chemical structure to identify unique carbon environments
    cot_instruction_6 = (
        "Sub-task 6: Analyze the chemical structure of the final product E to identify all unique carbon environments that will give rise to distinct 13C-NMR signals, "
        "considering symmetry and chemical equivalence."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, analyzing unique carbon environments in compound E, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    
    # Sub-task 7: Count the number of unique 13C-NMR signals and select the correct answer
    debate_instruction_7 = (
        "Sub-task 7: Based on the analysis of unique carbon environments, count the number of unique 13C-NMR signals and compare this number with the provided multiple-choice options (3, 11, 8, 6) to select the correct answer. "
        "Use debate among agents to ensure accuracy."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, counting unique 13C-NMR signals, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the number of unique 13C-NMR signals and select the correct multiple-choice answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct 13C-NMR signal count, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
