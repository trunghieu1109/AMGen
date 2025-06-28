async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Combine, Transform, and Prioritize Elements]
    
    [Objective] 
    - Integrate the combination and transformation of quantitative inputs with the evaluation and prioritization of elements.
    
    [Agent Collaborations]
    - Utilize SC_CoT, Reflexion, Debate, and CoT to achieve the stage objectives.
    
    [Examples]
    - Implement subtasks to combine and transform elements, prioritize them, and evaluate their significance.
    """
    
    # Sub-task 1: Combine and transform elements using SC_CoT
    cot_sc_instruction = "Sub-task 1: Combine and transform elements, considering all possible scenarios."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking1, answer1 = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers.append(answer1.content)
        thinkingmapping[answer1.content] = thinking1
        answermapping[answer1.content] = answer1
    
    most_common_answer = Counter(possible_answers).most_common(1)[0][0]
    thinking1 = thinkingmapping[most_common_answer]
    answer1 = answermapping[most_common_answer]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Analyze and Classify Elements]
    
    [Objective] 
    - Analyze elements to identify, evaluate, and classify their defining attributes, relationships, and functions.
    
    [Agent Collaborations]
    - Utilize CoT and SC_CoT to analyze and classify elements.
    
    [Examples]
    - Implement subtasks to analyze and classify elements based on their attributes and relationships.
    """
    
    # Sub-task 2: Analyze elements with CoT
    cot_instruction = "Sub-task 2: Analyze elements to identify key attributes and relationships."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing elements, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Compute Quantitative or Conditional Measure]
    
    [Objective] 
    - Calculate quantitative or conditional outputs by applying defined transformations, relationships, and criteria to the processed elements.
    
    [Agent Collaborations]
    - Utilize CoT and SC_CoT to compute quantitative measures.
    
    [Examples]
    - Implement subtasks to compute quantitative measures based on processed elements.
    """
    
    # Sub-task 3: Compute final output with Debate
    debate_instruction_3 = "Sub-task 3: Compute the final quantitative measure based on previous analyses."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            input_infos_3 = [taskInfo, thinking2, answer2]
            if r > 0:
                input_infos_3.extend(all_thinking3[r-1])
            thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make a final decision on the quantitative measure.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer