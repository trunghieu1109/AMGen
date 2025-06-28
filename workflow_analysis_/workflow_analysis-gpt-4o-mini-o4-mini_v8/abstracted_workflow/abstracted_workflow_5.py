async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Stage 1: Identify constrained combinations
    # Objective: Identify, generate, or select subsets or combinations of elements from a given set that satisfy one or more specified constraints or relational conditions.
    # Agent Collaborations: SC_CoT, CoT
    
    # Sub-task 1: Identify possible combinations with self-consistency
    cot_instruction = "Sub-task 1: Identify possible combinations of elements that satisfy constraints with context from [taskInfo]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {{cot_agent.id}, identifying combinations, thinking: {{thinking1.content}; answer: {{answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{thinking1.content}; answer - {{answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Validate combinations with self-consistency
    cot_sc_instruction = "Sub-task 2: Validate identified combinations and ensure consistency"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {{cot_agents[i].id}, validating combinations, thinking: {{thinking2.content}; answer: {{answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {{thinking2.content}; answer - {{answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # Stage 2: Compute adjusted aggregate measure
    # Objective: Transform, combine, and aggregate multiple input elements through defined operations to produce a single consolidated summary output.
    # Agent Collaborations: CoT, Reflexion
    
    # Sub-task 3: Compute aggregate measure with reflexion
    cot_reflect_instruction = "Sub-task 3: Compute aggregate measure based on validated combinations"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {{cot_agent.id}, computing aggregate measure, thinking: {{thinking3.content}; answer: {{answer3.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 
                                       "Review aggregate measure calculation and provide feedback.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {{critic_agent.id}, providing feedback, thinking: {{feedback.content}; answer: {{correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {{cot_agent.id}, refining aggregate measure, thinking: {{thinking3.content}; answer: {{answer3.content}")
    
    sub_tasks.append(f"Sub-task 3 output: thinking - {{thinking3.content}; answer - {{answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    # Stage 3: Derive primary variable
    # Objective: Compute one or more quantitative outputs by applying defined functional, arithmetic, or relational transformations to given input values and/or previously derived quantities.
    # Agent Collaborations: CoT, SC_CoT
    
    # Sub-task 4: Derive primary variable with self-consistency
    cot_sc_instruction = "Sub-task 4: Derive primary variable from aggregate measure"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking4, answer4 = await cot_agents[i]([taskInfo, thinking3, answer3], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {{cot_agents[i].id}, deriving primary variable, thinking: {{thinking4.content}; answer: {{answer4.content}")
        possible_answers.append(answer4.content)
        thinkingmapping[answer4.content] = thinking4
        answermapping[answer4.content] = answer4
    
    answer4 = Counter(possible_answers).most_common(1)[0][0]
    thinking4 = thinkingmapping[answer4]
    answer4 = answermapping[answer4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {{thinking4.content}; answer - {{answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    # Stage 4: Simplify ratio and aggregate components
    # Objective: Transform a quantitative relationship into its simplest integer-based components and compute an aggregate value from these components.
    # Agent Collaborations: Debate, Reflexion
    
    # Sub-task 5: Simplify and aggregate with debate
    debate_instruction_5 = "Sub-task 5: Simplify ratio and aggregate components"
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_5 = self.max_round
    
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], 
                                           debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            
            agents.append(f"Debate agent {{agent.id}, round {{r}, simplifying and aggregating, thinking: {{thinking5.content}; answer: {{answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                 "Sub-task 5: Make final decision on simplified and aggregated components.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing simplification and aggregation, thinking: {{thinking5.content}; answer: {{answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {{thinking5.content}; answer - {{answer5.content}")
    
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer