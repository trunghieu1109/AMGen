async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Stage 1: Derive Numerical Value
    # Objective: Compute a derived numerical value by applying the specified calculation procedure to input values.
    # Agent Collaborations: CoT, Reflexion
    cot_instruction = "Sub-task 1: Compute a derived numerical value by applying the specified calculation procedure to input values."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing derived numerical value, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    # Stage 2: Select Qualifying Elements
    # Objective: Select elements that satisfy defined criteria.
    # Agent Collaborations: Debate, CoT
    debate_instruction = "Sub-task 2: Select elements that satisfy defined criteria."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    
    all_thinking2 = [[] for _ in range(N_max)]
    all_answer2 = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], 
                                           debate_instruction, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting qualifying elements, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + all_thinking2[-1] + all_answer2[-1], 
                                                 "Sub-task 2: Make final decision on selected elements.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, selecting qualifying elements, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    # Stage 3: Classify Inputs Based on Attributes and Relationships
    # Objective: Categorize inputs into appropriate classes based on their attributes and relationships.
    # Agent Collaborations: CoT, SC_CoT
    cot_sc_instruction = "Sub-task 3: Categorize inputs into appropriate classes based on their attributes and relationships."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking3, answer3 = await cot_agents[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, categorizing inputs, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers.append(answer3.content)
        thinkingmapping[answer3.content] = thinking3
        answermapping[answer3.content] = answer3
    
    answer3 = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinkingmapping[answer3]
    answer3 = answermapping[answer3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    # Stage 4: Derive Target Value
    # Objective: Calculate a target output value by applying defined operations under specified conditions.
    # Agent Collaborations: Reflexion, SC_CoT
    cot_reflect_instruction = "Sub-task 4: Calculate a target output value by applying defined operations under specified conditions."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    
    thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, calculating target value, thinking: {thinking4.content}; answer: {answer4.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking4, answer4], 
                                       "Review target value calculation for accuracy and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining target value, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    # Stage 5: Compute Conditional Measure
    # Objective: Determine or adjust a quantitative measure based on specified conditions or transformation rules.
    # Agent Collaborations: Reflexion, CoT
    cot_reflect_instruction = "Sub-task 5: Determine or adjust a quantitative measure based on specified conditions or transformation rules."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    
    thinking5, answer5 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, computing conditional measure, thinking: {thinking5.content}; answer: {answer5.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking5, answer5], 
                                       "Review conditional measure computation for accuracy and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining conditional measure, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    # Stage 6: Assess Modification Impact
    # Objective: Evaluate the effect of a specified change on the input or system.
    # Agent Collaborations: SC_CoT, Reflexion
    cot_sc_instruction = "Sub-task 6: Evaluate the effect of a specified change on the input or system."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking6, answer6 = await cot_agents[i]([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, assessing modification impact, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers.append(answer6.content)
        thinkingmapping[answer6.content] = thinking6
        answermapping[answer6.content] = answer6
    
    answer6 = Counter(possible_answers).most_common(1)[0][0]
    thinking6 = thinkingmapping[answer6]
    answer6 = answermapping[answer6]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    
    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer