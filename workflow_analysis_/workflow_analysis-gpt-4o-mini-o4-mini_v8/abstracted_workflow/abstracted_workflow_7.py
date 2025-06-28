async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Identify Constrained Combinations]
    
    [Objective] 
    - Generate subsets of elements satisfying specified constraints.
    
    [Agent Collaborations]
    - Utilize SC_CoT and CoT to explore and identify valid combinations.
    
    [Examples]
    - Sub-task 1: Use CoT to analyze possible combinations.
    - Sub-task 2: Use SC_CoT to ensure consistency in identified combinations.
    """
    
    # Sub-task 1: Identify possible combinations using CoT
    cot_instruction = "Sub-task 1: Identify possible combinations of elements that satisfy constraints."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying combinations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Validate combinations using SC_CoT
    cot_sc_instruction = "Sub-task 2: Validate identified combinations for consistency."
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
    [Stage 2: Derive Primary Variable]
    
    [Objective] 
    - Compute quantitative outputs by applying defined transformations.
    
    [Agent Collaborations]
    - Use CoT and SC_CoT to derive and validate primary variables.
    
    [Examples]
    - Sub-task 3: Use CoT to derive primary variables.
    - Sub-task 4: Use SC_CoT to ensure consistency in derived variables.
    """
    
    # Sub-task 3: Derive primary variable using CoT
    cot_instruction = "Sub-task 3: Derive primary variable by applying transformations."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent([taskInfo, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, deriving primary variable, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Validate primary variable using SC_CoT
    cot_sc_instruction = "Sub-task 4: Validate derived primary variable for consistency."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking4, answer4 = await cot_agents[i]([taskInfo, thinking3, answer3], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, validating primary variable, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers.append(answer4.content)
        thinkingmapping[answer4.content] = thinking4
        answermapping[answer4.content] = answer4
    
    answer4 = Counter(possible_answers).most_common(1)[0][0]
    thinking4 = thinkingmapping[answer4]
    answer4 = answermapping[answer4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Determine Scalar Transformation Characteristics]
    
    [Objective] 
    - Analyze a scalar mapping to identify critical parameters.
    
    [Agent Collaborations]
    - Use CoT and SC_CoT to analyze and validate scalar transformations.
    
    [Examples]
    - Sub-task 5: Use CoT to analyze scalar transformations.
    - Sub-task 6: Use SC_CoT to ensure consistency in analysis.
    """
    
    # Sub-task 5: Analyze scalar transformation using CoT
    cot_instruction = "Sub-task 5: Analyze scalar transformation to identify critical parameters."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent([taskInfo, thinking4, answer4], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing scalar transformation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Validate scalar transformation analysis using SC_CoT
    cot_sc_instruction = "Sub-task 6: Validate scalar transformation analysis for consistency."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking6, answer6 = await cot_agents[i]([taskInfo, thinking5, answer5], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, validating scalar transformation analysis, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers.append(answer6.content)
        thinkingmapping[answer6.content] = thinking6
        answermapping[answer6.content] = answer6
    
    answer6 = Counter(possible_answers).most_common(1)[0][0]
    thinking6 = thinkingmapping[answer6]
    answer6 = answermapping[answer6]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer