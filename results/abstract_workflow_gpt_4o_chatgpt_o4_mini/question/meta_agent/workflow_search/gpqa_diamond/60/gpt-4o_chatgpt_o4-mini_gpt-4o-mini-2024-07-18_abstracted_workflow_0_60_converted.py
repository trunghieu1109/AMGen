async def forward_60(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Evaluate and Characterize Input Attributes
    # Sub-task 1: Identify and characterize the initial reactant, benzene, and the reagents HNO3 and H2SO4 used in the first reaction step.
    cot_instruction_1 = "Sub-task 1: Identify and characterize benzene and the reagents HNO3 and H2SO4 used in the first reaction step."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, characterizing benzene and reagents, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Characterize the product of the first reaction (nitration of benzene) and identify its structure.
    cot_instruction_2 = "Sub-task 2: Based on Sub-task 1 output, characterize the product of nitration of benzene."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, characterizing nitration product, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Identify and characterize the reagents Br2 and iron powder used in the second reaction step.
    cot_instruction_3 = "Sub-task 3: Identify and characterize the reagents Br2 and iron powder used in the second reaction step."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, characterizing Br2 and iron powder, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Characterize the product of the second reaction (bromination of the nitrobenzene) and identify its structure.
    cot_instruction_4 = "Sub-task 4: Based on Sub-task 2 and 3 outputs, characterize the product of bromination of nitrobenzene."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, characterizing bromination product, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Identify and characterize the reagents Pd/C and hydrogen used in the third reaction step.
    cot_instruction_5 = "Sub-task 5: Identify and characterize the reagents Pd/C and hydrogen used in the third reaction step."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, characterizing Pd/C and hydrogen, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Characterize the product of the third reaction (reduction of the bromo-nitrobenzene) and identify its structure.
    cot_instruction_6 = "Sub-task 6: Based on Sub-task 4 and 5 outputs, characterize the product of reduction of bromo-nitrobenzene."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, characterizing reduction product, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Sub-task 7: Identify and characterize the reagents NaNO2 and HBF4 used in the fourth reaction step.
    cot_instruction_7 = "Sub-task 7: Identify and characterize the reagents NaNO2 and HBF4 used in the fourth reaction step."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, characterizing NaNO2 and HBF4, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    # Sub-task 8: Characterize the product of the fourth reaction (diazotization and fluorination) and identify its structure.
    cot_instruction_8 = "Sub-task 8: Based on Sub-task 6 and 7 outputs, characterize the product of diazotization and fluorination."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking6, answer6, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, characterizing diazotization product, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Subtask 8 answer: ", sub_tasks[-1])

    # Sub-task 9: Identify and characterize the conditions of heating and the reagent anisole used in the final reaction step.
    cot_instruction_9 = "Sub-task 9: Identify and characterize the conditions of heating and the reagent anisole used in the final reaction step."
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking9, answer9 = await cot_agent_9([taskInfo], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, characterizing heating conditions and anisole, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Subtask 9 answer: ", sub_tasks[-1])

    # Sub-task 10: Characterize the final product of the reaction sequence and compare it with the given choices.
    cot_instruction_10 = "Sub-task 10: Based on Sub-task 8 and 9 outputs, characterize the final product and compare with given choices."
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking10, answer10 = await cot_agent_10([taskInfo, thinking8, answer8, thinking9, answer9], cot_instruction_10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_10.id}, characterizing final product, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    print("Subtask 10 answer: ", sub_tasks[-1])

    # Stage 1: Select the correct structure of the final product from the given choices based on the characterized transformations.
    cot_instruction_11 = "Sub-task 11: Select the correct structure of the final product from the given choices based on the characterized transformations."
    cot_agent_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking11, answer11 = await cot_agent_11([taskInfo, thinking10, answer10], cot_instruction_11, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_11.id}, selecting final product structure, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    print("Subtask 11 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking11, answer11, sub_tasks, agents)
    return final_answer