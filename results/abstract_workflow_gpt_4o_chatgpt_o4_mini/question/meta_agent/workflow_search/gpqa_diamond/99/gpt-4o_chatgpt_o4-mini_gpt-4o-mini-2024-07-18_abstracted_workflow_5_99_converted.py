async def forward_99(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    cot_instruction_1 = "Sub-task 1: Identify and characterize compound A (C3H6) and its initial properties."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying compound A, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_instruction_2 = "Sub-task 2: Understand the bromination reaction of compound A in the presence of carbon tetrachloride to form compound B."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, understanding bromination reaction, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])

    cot_instruction_3 = "Sub-task 3: Apply the reaction of compound B with alcoholic KOH to form compound C."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, applying reaction with alcoholic KOH, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    cot_instruction_4 = "Sub-task 4: Analyze the transformation of compound C when passed through a red-hot iron tube to form compound D."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, analyzing transformation to compound D, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])

    cot_instruction_5 = "Sub-task 5: Determine the reaction of compound D with a mixture of two strong acids to form compound E."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, determining reaction to form compound E, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    print("Subtask 5 answer: ", sub_tasks[-1])

    cot_instruction_6 = "Sub-task 6: Evaluate the formation of compound F from compound E in the presence of iron scrap and hydrochloric acid."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, evaluating formation of compound F, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    
    print("Subtask 6 answer: ", sub_tasks[-1])

    cot_instruction_7 = "Sub-task 7: Assess the reaction of compound F with nitrous acid to form compound G."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, assessing reaction to form compound G, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    
    print("Subtask 7 answer: ", sub_tasks[-1])

    cot_instruction_8 = "Sub-task 8: Analyze the reaction of compound G with sodium hydroxide to form compound H."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, analyzing reaction to form compound H, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    
    print("Subtask 8 answer: ", sub_tasks[-1])

    cot_instruction_9 = "Sub-task 9: Evaluate the properties of compound D, specifically its 1H NMR spectra, to verify if it gives two singlets."
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking4, answer4], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, evaluating 1H NMR spectra of compound D, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    
    print("Subtask 9 answer: ", sub_tasks[-1])

    cot_instruction_10 = "Sub-task 10: Assess the use of compound F in the synthesis of dyes."
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking10, answer10 = await cot_agent_10([taskInfo, thinking6, answer6], cot_instruction_10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_10.id}, assessing use of compound F in dye synthesis, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    
    print("Subtask 10 answer: ", sub_tasks[-1])

    cot_instruction_11 = "Sub-task 11: Determine if compound H gives a yellow color with the addition of ferric chloride solution."
    cot_agent_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking11, answer11 = await cot_agent_11([taskInfo, thinking8, answer8], cot_instruction_11, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_11.id}, determining color reaction of compound H, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    
    print("Subtask 11 answer: ", sub_tasks[-1])

    cot_instruction_12 = "Sub-task 12: Verify if compound C is a flammable gas."
    cot_agent_12 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking12, answer12 = await cot_agent_12([taskInfo, thinking3, answer3], cot_instruction_12, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_12.id}, verifying flammability of compound C, thinking: {thinking12.content}; answer: {answer12.content}")
    sub_tasks.append(f"Sub-task 12 output: thinking - {thinking12.content}; answer - {answer12.content}")
    
    print("Subtask 12 answer: ", sub_tasks[-1])

    debate_instruction_13 = "Sub-task 13: Select the incorrect statement about the products obtained in the reaction sequence based on the evaluations from previous subtasks."
    debate_agents_13 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_13 = self.max_round
    
    all_thinking13 = [[] for _ in range(N_max_13)]
    all_answer13 = [[] for _ in range(N_max_13)]
    
    for r in range(N_max_13):
        for i, agent in enumerate(debate_agents_13):
            if r == 0:
                thinking13, answer13 = await agent([taskInfo, thinking9, answer9, thinking10, answer10, thinking11, answer11, thinking12, answer12], 
                                           debate_instruction_13, r, is_sub_task=True)
            else:
                input_infos_13 = [taskInfo, thinking9, answer9, thinking10, answer10, thinking11, answer11, thinking12, answer12] + all_thinking13[r-1] + all_answer13[r-1]
                thinking13, answer13 = await agent(input_infos_13, debate_instruction_13, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, selecting incorrect statement, thinking: {thinking13.content}; answer: {answer13.content}")
            all_thinking13[r].append(thinking13)
            all_answer13[r].append(answer13)
    
    final_decision_agent_13 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking13, answer13 = await final_decision_agent_13([taskInfo] + all_thinking13[-1] + all_answer13[-1], 
                                                 "Sub-task 13: Make final decision on incorrect statement.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, selecting incorrect statement, thinking: {thinking13.content}; answer: {answer13.content}")
    sub_tasks.append(f"Sub-task 13 output: thinking - {thinking13.content}; answer - {answer13.content}")
    
    print("Subtask 13 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking13, answer13, sub_tasks, agents)
    return final_answer