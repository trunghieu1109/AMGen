async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Stage 1: Derive Numerical Value
    # Objective: Compute a derived numerical value from inputs using CoT and Reflexion.
    cot_instruction = "Sub-task 1: Consider/calculate all possible cases of deriving a numerical value from inputs."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, consider/calculate all possible scenarios, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    # Reflexion to refine the derived value
    cot_reflect_instruction = "Sub-task 2: Refine the derived numerical value based on initial calculations."
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1]
    thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, refining derived value, thinking: {thinking2.content}; answer: {answer2.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking2, answer2], 
                                       "Review the derived value and provide feedback.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining derived value, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")

    # Stage 2: Evaluate and Characterize Input Attributes
    # Objective: Analyze and interpret input attributes using SC_CoT and CoT.
    cot_sc_instruction = "Sub-task 3: Analyze and interpret input attributes to characterize them."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking3, answer3 = await cot_agents[i]([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, analyzing input attributes, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers.append(answer3.content)
        thinkingmapping[answer3.content] = thinking3
        answermapping[answer3.content] = answer3
    
    answer3 = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinkingmapping[answer3]
    answer3 = answermapping[answer3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")

    # Stage 3: Assess Consistency and Variation Significance
    # Objective: Analyze a collection to assess internal consistency and variation significance using SC_CoT and CoT.
    cot_sc_instruction = "Sub-task 4: Assess internal consistency and variation significance."
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking4, answer4 = await cot_agents[i]([taskInfo, thinking3, answer3], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, assessing consistency, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers.append(answer4.content)
        thinkingmapping[answer4.content] = thinking4
        answermapping[answer4.content] = answer4
    
    answer4 = Counter(possible_answers).most_common(1)[0][0]
    thinking4 = thinkingmapping[answer4]
    answer4 = answermapping[answer4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")

    # Stage 4: Evaluate and Determine Element Status
    # Objective: Evaluate elements to determine their qualification status or relative ordering using CoT and Debate.
    debate_instruction = "Sub-task 5: Evaluate elements to determine their qualification status."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], 
                                           debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking4, answer4] + all_thinking[r-1] + all_answer[r-1]
                thinking5, answer5 = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating elements, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking[r].append(thinking5)
            all_answer[r].append(answer5)
    
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], 
                                                 "Sub-task 5: Make final decision on element status.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, determining element status, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer