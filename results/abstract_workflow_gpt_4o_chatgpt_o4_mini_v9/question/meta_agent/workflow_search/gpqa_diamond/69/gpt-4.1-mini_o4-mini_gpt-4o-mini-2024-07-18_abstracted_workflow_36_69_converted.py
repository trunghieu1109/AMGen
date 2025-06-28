async def forward_69(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 0: Extract and characterize chemical species and relationships
    
    # Sub-task 1: Extract and identify chemical species and their relationships
    cot_instruction_1 = (
        "Sub-task 1: Extract and identify the chemical species involved (A, B, C, D, E, F, G, H) and their relationships from the query, "
        "including stoichiometric ratios and reaction sequences, to understand the system's components and transformations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting chemical species and relationships, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    
    # Sub-task 2: Characterize properties of acids F and G
    cot_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1 output, characterize the properties of acids F and G formed upon hydrolysis of C, "
        "noting that F is a strong acid and G is a weak acid, to aid in identifying the nature of C and its functional groups."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, characterizing acids F and G, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    
    # Sub-task 3: Analyze reaction between D and B forming H
    cot_instruction_3 = (
        "Sub-task 3: Based on Sub-task 1 output, analyze the reaction between D and B forming H in a 1:1 ratio and note that H is used as a solvent, "
        "to understand the nature of D and B and their possible chemical identities."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing reaction of D and B to form H, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    
    # Stage 1: Determine chemical identities of C and E
    
    # Sub-task 4: Determine chemical identity or plausible structure of product C
    cot_instruction_4 = (
        "Sub-task 4: Determine the chemical identity or plausible structure of product C based on its formation from solid A and 8 equivalents of gas B, "
        "its bright red color, and its hydrolysis behavior producing A, F, and G."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Self-Consistency Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    N_sc = self.max_sc
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N_sc):
        thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_4.id}, determining identity of C, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    counter_4 = Counter(possible_answers_4)
    best_answer_4 = counter_4.most_common(1)[0][0]
    thinking4 = thinkingmapping_4[best_answer_4]
    answer4 = answermapping_4[best_answer_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    
    # Sub-task 5: Determine chemical identity or plausible structure of product E
    debate_instruction_5 = (
        "Sub-task 5: Determine the chemical identity or plausible structure of product E formed by reaction of C with 2 equivalents of gas D, "
        "considering E is extremely hazardous, to understand its molecular structure for symmetry analysis."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
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
            agents.append(f"Debate agent {agent.id}, round {r}, determining identity of E, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the chemical identity of E.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding identity of E, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    
    # Stage 2: Analyze molecular symmetry of E and select correct symmetry group
    
    # Sub-task 6: Analyze molecular structure of E to identify symmetry elements and classify molecular symmetry group
    cot_instruction_6 = (
        "Sub-task 6: Analyze the molecular structure of product E to identify its symmetry elements and classify its molecular symmetry group, "
        "using the information about E's formation and chemical nature derived from previous subtasks."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, analyzing molecular symmetry of E, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    
    # Sub-task 7: Compare identified molecular symmetry group of E with provided choices
    cot_instruction_7 = (
        "Sub-task 7: Compare the identified molecular symmetry group of E with the provided choices (C2, C2v, D4h, D∞h) to select the correct answer."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, selecting correct molecular symmetry group, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
