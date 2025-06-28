async def forward_132(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Calculate moles and volume
    cot_instruction_1 = "Sub-task 1: Calculate the number of moles of KH2PO4 in the solution using its given mass and molar mass."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculating moles of KH2PO4, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_instruction_2 = "Sub-task 2: Calculate the number of moles of Na2HPO4●2H2O in the solution using its given mass and molar mass."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, calculating moles of Na2HPO4●2H2O, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    cot_instruction_3 = "Sub-task 3: Determine the total volume of the solution in liters to facilitate concentration calculations."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, determining total volume, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Calculate initial concentrations and analyze dissociation
    cot_instruction_4 = "Sub-task 4: Calculate the initial concentration of KH2PO4 in the solution using the moles calculated in subtask_1 and the volume from subtask_3."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating initial concentration of KH2PO4, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    cot_instruction_5 = "Sub-task 5: Calculate the initial concentration of Na2HPO4 in the solution using the moles calculated in subtask_2 and the volume from subtask_3."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating initial concentration of Na2HPO4, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    cot_instruction_6 = "Sub-task 6: Analyze the dissociation of KH2PO4 and Na2HPO4 in water to determine the species present in the solution."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, analyzing dissociation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    cot_instruction_7 = "Sub-task 7: Apply the equilibrium constants (Ka1, Ka2, Ka3) for H3PO4 to determine the concentration of orthophosphate ions in the solution."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, applying equilibrium constants, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer