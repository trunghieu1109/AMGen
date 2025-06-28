async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Stage 1: Classify inputs based on attributes and relationships
    # Objective: Analyze input entities and assign categories based on attributes and relationships
    # Agent Collaborations: CoT, SC_CoT
    
    # Sub-task 1: Classify inputs using CoT
    cot_instruction = "Sub-task 1: Classify inputs based on attributes and relationships"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, classifying inputs, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Evaluate classification consistency using SC_CoT
    cot_sc_instruction = "Sub-task 2: Evaluate classification consistency"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, evaluating consistency, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Evaluate and characterize input attributes
    # Objective: Analyze and interpret input attributes and relationships
    # Agent Collaborations: SC_CoT, CoT
    
    # Sub-task 3: Analyze attributes with SC_CoT
    cot_sc_instruction = "Sub-task 3: Analyze and interpret input attributes"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking3, answer3 = await cot_agents[i]([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, analyzing attributes, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers.append(answer3.content)
        thinkingmapping[answer3.content] = thinking3
        answermapping[answer3.content] = answer3
    
    answer3 = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinkingmapping[answer3]
    answer3 = answermapping[answer3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 3: Analyze relationships and contributing factors
    # Objective: Analyze relationships and contributing factors within the system
    # Agent Collaborations: Debate, CoT
    
    # Sub-task 4: Debate relationships and factors
    debate_instruction = "Sub-task 4: Debate relationships and contributing factors"
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
            
            agents.append(f"Debate agent {agent.id}, round {r}, debating relationships, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], 
                                                 "Sub-task 4: Make final decision on relationships.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, deciding on relationships, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 4: Derive output
    # Objective: Derive output by applying defined transformations to inputs
    # Agent Collaborations: SC_CoT, CoT
    
    # Sub-task 5: Derive output using SC_CoT
    cot_sc_instruction = "Sub-task 5: Derive output by applying transformations"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking5, answer5 = await cot_agents[i]([taskInfo, thinking4, answer4], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, deriving output, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers.append(answer5.content)
        thinkingmapping[answer5.content] = thinking5
        answermapping[answer5.content] = answer5
    
    answer5 = Counter(possible_answers).most_common(1)[0][0]
    thinking5 = thinkingmapping[answer5]
    answer5 = answermapping[answer5]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Stage 5: Select elements by condition satisfaction
    # Objective: Identify and select elements satisfying specified conditions
    # Agent Collaborations: Debate, SC_CoT
    
    # Sub-task 6: Debate condition satisfaction
    debate_instruction = "Sub-task 6: Debate condition satisfaction"
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    
    all_thinking6 = [[] for _ in range(N_max)]
    all_answer6 = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], 
                                           debate_instruction, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, debating condition satisfaction, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], 
                                                 "Sub-task 6: Make final decision on condition satisfaction.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, deciding on condition satisfaction, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
