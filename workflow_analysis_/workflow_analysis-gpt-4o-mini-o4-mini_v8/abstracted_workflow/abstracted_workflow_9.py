async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Determine Scalar Transformation Characteristics]
    
    [Objective] 
    - Analyze a scalar relation to identify and characterize its critical parameters.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT) and Self-Consistency Chain-of-Thought (SC_CoT) are used to explore and validate the characteristics of scalar transformations.
    
    [Examples]
    - Sub-task 1: Analyze the scalar relation using CoT to identify key parameters.
    - Sub-task 2: Validate the identified parameters using SC_CoT to ensure consistency.
    """
    
    # Sub-task 1: Analyze scalar relation with CoT
    cot_instruction = "Sub-task 1: Analyze the scalar relation to identify key parameters with context from [taskInfo]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing scalar relation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Validate parameters with SC_CoT
    cot_sc_instruction = "Sub-task 2: Validate the identified parameters using SC_CoT"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, validating parameters, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Compose Mapping Functions]
    
    [Objective] 
    - Define composite mappings that systematically relate inputs to outputs.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT) and Debate are used to explore and refine mapping functions.
    
    [Examples]
    - Sub-task 3: Define initial mapping functions using CoT.
    - Sub-task 4: Refine mappings through Debate to ensure robustness.
    """
    
    # Sub-task 3: Define mapping functions with CoT
    cot_instruction = "Sub-task 3: Define initial mapping functions with context from [taskInfo]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, defining mapping functions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Refine mappings with Debate
    debate_instruction = "Sub-task 4: Refine mapping functions through Debate"
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
            
            agents.append(f"Debate agent {agent.id}, round {r}, refining mappings, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent([taskInfo] + all_thinking4[-1] + all_answer4[-1], 
                                                 "Sub-task 4: Make final decision on mappings.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, refining mappings, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Compute Adjusted Aggregate Measure]
    
    [Objective] 
    - Transform and combine elements to produce a single consolidated summary.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT) and Reflexion are used to compute and refine the aggregate measure.
    
    [Examples]
    - Sub-task 5: Compute initial aggregate measure using CoT.
    - Sub-task 6: Refine the measure through Reflexion to ensure accuracy.
    """
    
    # Sub-task 5: Compute aggregate measure with CoT
    cot_instruction = "Sub-task 5: Compute initial aggregate measure with context from [taskInfo]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing aggregate measure, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Refine measure with Reflexion
    cot_reflect_instruction = "Sub-task 6: Refine aggregate measure through Reflexion"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5]
    
    thinking6, answer6 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, refining measure, thinking: {thinking6.content}; answer: {answer6.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking6, answer6], 
                                       "Review aggregate measure for accuracy and provide limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining measure, thinking: {thinking6.content}; answer: {answer6.content}")
    
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    
    print("Subtask 6 answer: ", sub_tasks[-1])
    
    """
    [Stage 4: Determine Threshold Parameter]
    
    [Objective] 
    - Identify and quantify configurations to determine relevant threshold parameters.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT) and Self-Consistency Chain-of-Thought (SC_CoT) are used to explore and validate threshold parameters.
    
    [Examples]
    - Sub-task 7: Identify potential threshold parameters using CoT.
    - Sub-task 8: Validate the parameters using SC_CoT to ensure consistency.
    """
    
    # Sub-task 7: Identify threshold parameters with CoT
    cot_instruction = "Sub-task 7: Identify potential threshold parameters with context from [taskInfo]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying threshold parameters, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    
    print("Subtask 7 answer: ", sub_tasks[-1])

    # Sub-task 8: Validate parameters with SC_CoT
    cot_sc_instruction = "Sub-task 8: Validate the threshold parameters using SC_CoT"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking8, answer8 = await cot_agents[i]([taskInfo, thinking7, answer7], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, validating threshold parameters, thinking: {thinking8.content}; answer: {answer8.content}")
        possible_answers.append(answer8.content)
        thinkingmapping[answer8.content] = thinking8
        answermapping[answer8.content] = answer8
    
    answer8 = Counter(possible_answers).most_common(1)[0][0]
    thinking8 = thinkingmapping[answer8]
    answer8 = answermapping[answer8]
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    
    print("Subtask 8 answer: ", sub_tasks[-1])

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer
