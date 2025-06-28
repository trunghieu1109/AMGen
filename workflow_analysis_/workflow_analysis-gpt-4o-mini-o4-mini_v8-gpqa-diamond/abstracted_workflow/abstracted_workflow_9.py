async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Stage 1: Assess consistency and variation significance
    # Objective: Analyze internal consistency and significance of variations in the input collection.
    # Agent Collaborations: SC_CoT, CoT
    
    # Sub-task 1: Analyze consistency with CoT
    cot_instruction = "Sub-task 1: Analyze internal consistency of the input collection."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing consistency, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Analyze variation significance with SC_CoT
    cot_sc_instruction = "Sub-task 2: Analyze significance of variations based on Sub-task 1 output."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, analyzing variation significance, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # Stage 2: Extract defining features
    # Objective: Identify and isolate essential components and attributes of the input dataset.
    # Agent Collaborations: CoT, Debate
    
    # Sub-task 3: Extract features with CoT
    cot_instruction = "Sub-task 3: Identify essential components of the input dataset."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent([taskInfo, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting features, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Debate on feature extraction
    debate_instruction = "Sub-task 4: Debate on the extracted features and refine the selection."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    
    all_thinking4 = [[] for _ in range(N_max)]
    all_answer4 = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], 
                                           debate_instruction, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, debating features, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent([taskInfo] + all_thinking4[-1] + all_answer4[-1], 
                                                 "Sub-task 4: Make final decision on feature selection.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing features, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    # Stage 3: Derive quantitative measure
    # Objective: Compute meaningful numeric metrics from input data.
    # Agent Collaborations: CoT, SC_CoT
    
    # Sub-task 5: Derive metrics with CoT
    cot_instruction = "Sub-task 5: Compute numeric metrics from the input data."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent([taskInfo, thinking4, answer4], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, deriving metrics, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Validate metrics with SC_CoT
    cot_sc_instruction = "Sub-task 6: Validate the derived metrics for consistency."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking6, answer6 = await cot_agents[i]([taskInfo, thinking5, answer5], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, validating metrics, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers.append(answer6.content)
        thinkingmapping[answer6.content] = thinking6
        answermapping[answer6.content] = answer6
    
    answer6 = Counter(possible_answers).most_common(1)[0][0]
    thinking6 = thinkingmapping[answer6]
    answer6 = answermapping[answer6]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    
    print("Subtask 6 answer: ", sub_tasks[-1])
    
    # Stage 4: Evaluate and characterize input attributes
    # Objective: Analyze and interpret defining attributes and behaviors of inputs.
    # Agent Collaborations: SC_CoT, CoT
    
    # Sub-task 7: Characterize attributes with CoT
    cot_instruction = "Sub-task 7: Characterize defining attributes of the input."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent([taskInfo, thinking6, answer6], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, characterizing attributes, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    
    print("Subtask 7 answer: ", sub_tasks[-1])

    # Sub-task 8: Validate attribute characterization with SC_CoT
    cot_sc_instruction = "Sub-task 8: Validate the characterization of attributes."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking8, answer8 = await cot_agents[i]([taskInfo, thinking7, answer7], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, validating attribute characterization, thinking: {thinking8.content}; answer: {answer8.content}")
        possible_answers.append(answer8.content)
        thinkingmapping[answer8.content] = thinking8
        answermapping[answer8.content] = answer8
    
    answer8 = Counter(possible_answers).most_common(1)[0][0]
    thinking8 = thinkingmapping[answer8]
    answer8 = answermapping[answer8]
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    
    print("Subtask 8 answer: ", sub_tasks[-1])
    
    # Stage 5: Evaluate and select candidate(s)
    # Objective: Evaluate elements against defined criteria and select those that best satisfy requirements.
    # Agent Collaborations: Debate, SC_CoT
    
    # Sub-task 9: Debate on candidate selection
    debate_instruction = "Sub-task 9: Debate on the selection of candidates based on defined criteria."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    
    all_thinking9 = [[] for _ in range(N_max)]
    all_answer9 = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking9, answer9 = await agent([taskInfo, thinking8, answer8], 
                                           debate_instruction, r, is_sub_task=True)
            else:
                input_infos_9 = [taskInfo, thinking8, answer8] + all_thinking9[r-1] + all_answer9[r-1]
                thinking9, answer9 = await agent(input_infos_9, debate_instruction, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, debating candidate selection, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)
    
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent([taskInfo] + all_thinking9[-1] + all_answer9[-1], 
                                                 "Sub-task 9: Make final decision on candidate selection.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing candidate selection, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    
    print("Subtask 9 answer: ", sub_tasks[-1])

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer