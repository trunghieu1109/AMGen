async def forward_132(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Calculate moles and convert volume
    # Sub-task 1: Calculate moles of KH2PO4 and Na2HPO4·2H2O using given masses and molecular weights
    cot_instruction_1 = (
        "Sub-task 1: Calculate the number of moles of KH2PO4 and Na2HPO4·2H2O in the 200.00 cm3 solution "
        "using their given masses and molecular weights."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculating moles, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Convert volume from cm3 to liters
    cot_instruction_2 = (
        "Sub-task 2: Convert the solution volume from 200.00 cm3 to liters to prepare for concentration calculations."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, converting volume, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Determine initial concentrations
    # Sub-task 3: Calculate initial concentrations of phosphate species by dividing moles by volume in liters
    cot_instruction_3 = (
        "Sub-task 3: Determine the initial concentrations of phosphate species from KH2PO4 and Na2HPO4·2H2O "
        "by dividing moles by volume in liters."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating initial concentrations, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Analyze dissociation equilibria and classify phosphate species
    # Sub-task 4: Analyze dissociation equilibria of phosphoric acid using Ka values
    cot_instruction_4 = (
        "Sub-task 4: Analyze the dissociation equilibria of phosphoric acid (H3PO4) using the given Ka1, Ka2, and Ka3 values "
        "to understand the distribution of phosphate species at pH 7."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, analyzing dissociation equilibria, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Classify phosphate species present based on dissociation constants and initial concentrations
    cot_instruction_5 = (
        "Sub-task 5: Classify the phosphate species present in solution (H3PO4, H2PO4-, HPO4^2-, PO4^3-) "
        "based on the dissociation constants and initial concentrations from KH2PO4 and Na2HPO4·2H2O."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, classifying phosphate species, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 3: Calculate orthophosphate ion concentration and compare with choices
    # Sub-task 6: Calculate concentration of orthophosphate ions applying equilibrium expressions and mass balance
    cot_sc_instruction_6 = (
        "Sub-task 6: Calculate the concentration of orthophosphate ions in the solution by applying equilibrium expressions "
        "and mass balance using the initial concentrations and Ka values."
    )
    N = self.max_sc
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6 = []
    thinkingmapping_6 = {}
    answermapping_6 = {}
    for i in range(N):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking5, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, calculating orthophosphate concentration, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6.content)
        thinkingmapping_6[answer6.content] = thinking6
        answermapping_6[answer6.content] = answer6
    most_common_answer_6 = Counter(possible_answers_6).most_common(1)[0][0]
    thinking6 = thinkingmapping_6[most_common_answer_6]
    answer6 = answermapping_6[most_common_answer_6]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Compare calculated orthophosphate concentration with provided multiple-choice answers
    debate_instruction_7 = (
        "Sub-task 7: Compare the calculated orthophosphate ion concentration with the provided multiple-choice answers "
        "to identify the correct option."
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
            agents.append(f"Debate agent {agent.id}, round {r}, comparing with choices, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the correct multiple-choice answer.", is_sub_task=True)
    agents.append(f"Final Decision agent on multiple-choice selection, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
