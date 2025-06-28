async def forward(self, taskInfo):
    
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Identify and list all the given data points and their relationships
    cot_instruction = "Sub-task 1: Identify and list all the given data points and their relationships, such as the number of residents owning each item and combinations of items."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, identifying data points, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze the relationship between the number of residents owning exactly two items and those owning exactly three items
    cot_sc_instruction = "Sub-task 2: Analyze the relationship between the number of residents owning exactly two items and those owning exactly three items."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, analyzing relationships, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 3: Calculate the number of residents who own at least one of the three items using the principle of inclusion-exclusion
    cot_reflect_instruction = "Sub-task 3: Calculate the number of residents who own at least one of the three items (diamond ring, golf clubs, garden spade) using the principle of inclusion-exclusion."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, calculating residents owning at least one item, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], 
                                       "Critically evaluate the inclusion-exclusion calculation and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining calculation, thinking: {thinking3.content}; answer: {answer3.content}')
    
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 4: Determine the number of residents who own all three items
    cot_reflect_instruction = "Sub-task 4: Determine the number of residents who own all three items (diamond ring, golf clubs, garden spade) using the data on residents owning exactly two and three items."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking2, answer2, thinking3, answer3]
    
    thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, determining residents owning all three items, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking4, answer4], 
                                       "Review the calculation of residents owning all three items and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining calculation, thinking: {thinking4.content}; answer: {answer4.content}')
    
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 5: Calculate the number of residents who own all four items
    debate_instruction_5 = "Sub-task 5: Calculate the number of residents who own all four items by considering the total number of residents and those owning all three items."
    debate_agents_5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_5 = self.max_round
    
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = agent([taskInfo, thinking4, answer4], 
                                           debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            
            agents.append(f'Debate agent {agent.id}, round {r}, calculating residents owning all four items, thinking: {thinking5.content}; answer: {answer5.content}')
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    
    final_decision_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                 "Sub-task 5: Make final decision on the number of residents owning all four items.", 
                                                 is_sub_task=True)
    agents.append(f'Final Decision agent, calculating final number of residents owning all four items, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer