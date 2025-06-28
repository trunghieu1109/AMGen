async def forward_157(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    cot_instruction_1 = "Sub-task 1: Evaluate the characteristics of the transcription factor in its inactive and active states, focusing on the role of phosphorylation and dimerization."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, evaluating transcription factor states, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_instruction_2 = "Sub-task 2: Characterize the missense mutation X in the transactivation domain and its recessive loss-of-function nature."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, characterizing mutation X, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])

    cot_instruction_3 = "Sub-task 3: Characterize the heterozygous mutation Y in the dimerization domain and its dominant-negative nature."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, characterizing mutation Y, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    cot_instruction_4 = "Sub-task 4: Apply the concept of phosphorylation to understand how it activates the transcription factor and leads to dimerization."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, applying phosphorylation concept, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])

    cot_instruction_5 = "Sub-task 5: Apply the concept of dimerization to understand how mutation Y affects the transcription factors ability to dimerize."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, applying dimerization concept, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    print("Subtask 5 answer: ", sub_tasks[-1])

    cot_instruction_6 = "Sub-task 6: Extract the defining features of a dominant-negative mutation and how it can interfere with the function of the wild-type allele."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking3, answer3, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, extracting dominant-negative features, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    
    print("Subtask 6 answer: ", sub_tasks[-1])

    cot_instruction_7 = "Sub-task 7: Extract the defining features of a recessive loss-of-function mutation and how it affects the transcription factors activity."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking2, answer2], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, extracting recessive loss-of-function features, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    
    print("Subtask 7 answer: ", sub_tasks[-1])

    cot_instruction_8 = "Sub-task 8: Assess the impact of mutation Y on the transcription factors ability to dimerize and its overall function in gene transcription."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking6, answer6], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, assessing impact of mutation Y, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    
    print("Subtask 8 answer: ", sub_tasks[-1])

    cot_instruction_9 = "Sub-task 9: Assess the impact of mutation X on the transcription factors function in the context of a heterozygous state."
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking7, answer7], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, assessing impact of mutation X, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    
    print("Subtask 9 answer: ", sub_tasks[-1])

    debate_instruction_10 = "Sub-task 10: Select the most likely molecular phenotype from the given options based on the analysis of mutations X and Y."
    debate_agents_10 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_10 = self.max_round
    
    all_thinking10 = [[] for _ in range(N_max_10)]
    all_answer10 = [[] for _ in range(N_max_10)]
    
    for r in range(N_max_10):
        for i, agent in enumerate(debate_agents_10):
            if r == 0:
                thinking10, answer10 = await agent([taskInfo, thinking8, answer8, thinking9, answer9], 
                                           debate_instruction_10, r, is_sub_task=True)
            else:
                input_infos_10 = [taskInfo, thinking8, answer8, thinking9, answer9] + all_thinking10[r-1] + all_answer10[r-1]
                thinking10, answer10 = await agent(input_infos_10, debate_instruction_10, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, selecting molecular phenotype, thinking: {thinking10.content}; answer: {answer10.content}")
            all_thinking10[r].append(thinking10)
            all_answer10[r].append(answer10)
    
    final_decision_agent_10 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking10, answer10 = await final_decision_agent_10([taskInfo] + all_thinking10[-1] + all_answer10[-1], 
                                                 "Sub-task 10: Make final decision on molecular phenotype.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, selecting molecular phenotype, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    
    print("Subtask 10 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer