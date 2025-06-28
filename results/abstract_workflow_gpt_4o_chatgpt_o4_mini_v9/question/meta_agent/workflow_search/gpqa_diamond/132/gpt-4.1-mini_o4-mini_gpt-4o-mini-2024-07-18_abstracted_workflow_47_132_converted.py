async def forward_132(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 0: Calculate moles, concentrations, and identify species
    
    # Sub-task 1: Calculate moles of KH2PO4 and Na2HPO4·2H2O
    cot_instruction_1 = (
        "Sub-task 1: Calculate the number of moles of KH2PO4 and Na2HPO4·2H2O in the solution "
        "using their given masses (1.00 g each) and molecular weights (136.09 g/mol for KH2PO4 and 177.99 g/mol for Na2HPO4·2H2O). "
        "This is essential for further concentration calculations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculating moles, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    
    # Sub-task 2: Calculate molar concentrations of KH2PO4 and Na2HPO4·2H2O in 0.200 L solution
    cot_instruction_2 = (
        "Sub-task 2: Determine the molar concentrations of KH2PO4 and Na2HPO4·2H2O in the 200.00 cm³ (0.200 L) solution "
        "by dividing the moles calculated in Sub-task 1 by the solution volume in liters. "
        "This provides the initial concentrations of the phosphate species before equilibrium."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, calculating molar concentrations, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    
    # Sub-task 3: Identify phosphate species and dissociation equilibria
    cot_instruction_3 = (
        "Sub-task 3: Identify the phosphate species present in the solution from KH2PO4 and Na2HPO4·2H2O "
        "and their corresponding dissociation equilibria based on the given Ka1, Ka2, and Ka3 values for H3PO4. "
        "Clarify which phosphate ions contribute to the orthophosphate ion concentration and how acid-base equilibria relate."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, identifying species and equilibria, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    
    # Stage 1: Setup equilibrium expressions and calculate equilibrium concentrations
    
    # Sub-task 4: Setup equilibrium expressions using Ka values and initial concentrations
    cot_instruction_4 = (
        "Sub-task 4: Set up the equilibrium expressions for the dissociation of phosphate species in solution "
        "using the given Ka values (Ka1 = 7.5x10^-3, Ka2 = 6.2x10^-8, Ka3 = 1.8x10^-12) and initial concentrations from Sub-task 2. "
        "Write chemical equilibria and expressions relating species concentrations at equilibrium."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, setting up equilibrium expressions, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    
    # Sub-task 5: Calculate equilibrium concentrations of phosphate species including orthophosphate ions
    debate_instruction_5 = (
        "Sub-task 5: Based on the output of Sub-task 4, calculate the equilibrium concentrations of all relevant phosphate species, "
        "especially the orthophosphate ions (PO4^3-), by solving the system of equilibrium equations derived. "
        "Use initial concentrations and Ka values to find final species distribution."
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
            agents.append(f"Debate agent {agent.id}, round {r}, calculating equilibrium concentrations, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on equilibrium concentrations of phosphate species.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating equilibrium concentrations, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    
    # Sub-task 6: Determine total concentration of orthophosphate ions in solution
    cot_instruction_6 = (
        "Sub-task 6: Determine the total concentration of orthophosphate ions in the solution by summing the concentrations "
        "of all species that contribute to orthophosphate ion concentration (primarily PO4^3-). "
        "Compare this final concentration against the provided answer choices."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, determining total orthophosphate concentration, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
