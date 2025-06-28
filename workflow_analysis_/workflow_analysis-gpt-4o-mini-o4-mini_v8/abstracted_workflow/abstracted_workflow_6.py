async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Identify Constrained Combinations]
    
    [Objective] 
    - Identify combinations of elements that satisfy specified constraints.
    
    [Agent Collaborations]
    - Use SC_CoT and CoT to explore and identify valid combinations.
    
    [Examples]
    - Sub-task 1: Use CoT to analyze potential combinations.
    - Sub-task 2: Use SC_CoT to ensure consistency in identified combinations.
    """
    
    # Sub-task 1: Identify potential combinations using CoT
    cot_instruction = "Sub-task 1: Identify potential combinations of elements that satisfy constraints."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying combinations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Ensure consistency in identified combinations using SC_CoT
    cot_sc_instruction = "Sub-task 2: Validate and ensure consistency in identified combinations."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, validating combinations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Determine Threshold Parameter]
    
    [Objective] 
    - Quantify configurations that satisfy specified constraints and determine relevant threshold parameters.
    
    [Agent Collaborations]
    - Use CoT and SC_CoT to quantify and determine thresholds.
    
    [Examples]
    - Sub-task 3: Use CoT to quantify configurations.
    - Sub-task 4: Use SC_CoT to determine threshold parameters.
    """
    
    # Sub-task 3: Quantify configurations using CoT
    cot_instruction = "Sub-task 3: Quantify configurations that satisfy constraints."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent([taskInfo, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, quantifying configurations, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Determine threshold parameters using SC_CoT
    cot_sc_instruction = "Sub-task 4: Determine threshold parameters for configurations."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking4, answer4 = await cot_agents[i]([taskInfo, thinking3, answer3], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, determining thresholds, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers.append(answer4.content)
        thinkingmapping[answer4.content] = thinking4
        answermapping[answer4.content] = answer4
    
    answer4 = Counter(possible_answers).most_common(1)[0][0]
    thinking4 = thinkingmapping[answer4]
    answer4 = answermapping[answer4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Compute Adjusted Aggregate Measure]
    
    [Objective] 
    - Aggregate and summarize input elements into a consolidated measure.
    
    [Agent Collaborations]
    - Use CoT and Reflexion to compute and refine the aggregate measure.
    
    [Examples]
    - Sub-task 5: Use CoT to compute initial aggregate measure.
    - Sub-task 6: Use Reflexion to refine and validate the measure.
    """
    
    # Sub-task 5: Compute initial aggregate measure using CoT
    cot_instruction = "Sub-task 5: Compute initial aggregate measure from configurations."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent([taskInfo, thinking4, answer4], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing aggregate measure, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Refine and validate aggregate measure using Reflexion
    cot_reflect_instruction = "Sub-task 6: Refine and validate aggregate measure."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking5, answer5]
    
    thinking6, answer6 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, refining aggregate measure, thinking: {thinking6.content}; answer: {answer6.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking6, answer6], 
                                       "Review and validate the aggregate measure.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining aggregate measure, thinking: {thinking6.content}; answer: {answer6.content}")
    
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer