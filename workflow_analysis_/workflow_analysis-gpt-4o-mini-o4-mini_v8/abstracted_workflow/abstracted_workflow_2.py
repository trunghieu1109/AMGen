async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Stage 1: Identify constrained combinations
    # Objective: Identify subsets or combinations that satisfy specified constraints
    # Agent Collaborations: SC_CoT, CoT
    cot_instruction = "Sub-task 1: Consider/calculate all possible cases of constrained combinations with context ...."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, consider/calculate all possible scenarios of constrained combinations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    # Stage 2: Compose mapping functions
    # Objective: Define composite mappings to systematically relate inputs to outputs
    # Agent Collaborations: CoT, Debate
    debate_instruction = "Sub-task 2: Define composite mappings and debate their effectiveness"
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], 
                                           debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking1, answer1] + all_thinking[r-1] + all_answer[r-1]
                thinking2, answer2 = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, defining composite mappings, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking[r].append(thinking2)
            all_answer[r].append(answer2)
    
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], 
                                                 "Sub-task 2: Make final decision on composite mappings.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, deciding on composite mappings, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    # Stage 3: Compute adjusted aggregate measure
    # Objective: Compute an adjusted aggregate measure from multiple inputs
    # Agent Collaborations: CoT, Reflexion
    cot_reflect_instruction = "Sub-task 3: Compute adjusted aggregate measure based on previous outputs"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, computing adjusted aggregate measure, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 
                                       "please review the adjusted aggregate measure and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining adjusted aggregate measure, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    # Stage 4: Validate transformation output
    # Objective: Validate the transformation output against governing conditions or constraints
    # Agent Collaborations: CoT, Reflexion
    cot_reflect_instruction = "Sub-task 4: Validate transformation output against constraints"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    
    thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, validating transformation output, thinking: {thinking4.content}; answer: {answer4.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking4, answer4], 
                                       "please review the transformation validation and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining transformation validation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    # Stage 5: Derive primary variable
    # Objective: Compute primary quantitative outputs via defined functional or arithmetic transformations
    # Agent Collaborations: CoT, SC_CoT
    cot_sc_instruction = "Sub-task 5: Derive primary variable using previous outputs"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking5, answer5 = await cot_agents[i]([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, deriving primary variable, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers.append(answer5.content)
        thinkingmapping[answer5.content] = thinking5
        answermapping[answer5.content] = answer5
    
    answer5 = Counter(possible_answers).most_common(1)[0][0]
    thinking5 = thinkingmapping[answer5]
    answer5 = answermapping[answer5]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer