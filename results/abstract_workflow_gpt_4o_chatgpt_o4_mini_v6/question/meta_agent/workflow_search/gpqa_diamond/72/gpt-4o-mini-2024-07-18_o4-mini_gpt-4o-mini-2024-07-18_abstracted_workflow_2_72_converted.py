async def forward_72(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    cot_instruction = "Sub-task 1: Identify the masses and speeds of the two astronauts, as well as the relevant physical principles (relativity) that will be used to calculate relative speed and total energy."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying masses and speeds, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Classify the types of calculations needed: one for relative speed using the relativistic velocity addition formula and another for total energy using the relativistic energy formula."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, classifying calculations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    cot_instruction3 = "Sub-task 3: Calculate the relative speed of the two astronauts using the relativistic velocity addition formula: v_rel = (v1 + v2) / (1 + (v1 * v2 / c^2))."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, calculating relative speed, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    cot_instruction4 = "Sub-task 4: Calculate the total energy of the system using the formula E = mc^2 / sqrt(1 - (v^2/c^2)) for each astronaut and sum their energies."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking2, answer2], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, calculating total energy, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    cot_instruction5 = "Sub-task 5: Evaluate the results of the relative speed calculation and ensure it is less than the speed of light (c)."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent5([taskInfo, thinking3, answer3], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, evaluating relative speed, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    cot_instruction6 = "Sub-task 6: Evaluate the total energy results to ensure they are consistent with relativistic energy principles and check against the provided choices."
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent6([taskInfo, thinking4, answer4], cot_instruction6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, evaluating total energy, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    debate_instruction = "Sub-task 7: Compare the calculated relative speed and total energy with the provided choices to determine the correct answer."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking5, answer5, thinking6, answer6], 
                                           debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking5, answer5, thinking6, answer6] + all_thinking[r-1] + all_answer[r-1]
                thinking7, answer7 = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing results, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking[r].append(thinking7)
            all_answer[r].append(answer7)
    
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_final, answer_final = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], 
                                                 "Sub-task 7: Make final decision on the correct answer.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, determining correct answer, thinking: {thinking_final.content}; answer: {answer_final.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking_final.content}; answer - {answer_final.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_final, answer_final, sub_tasks, agents)
    return final_answer