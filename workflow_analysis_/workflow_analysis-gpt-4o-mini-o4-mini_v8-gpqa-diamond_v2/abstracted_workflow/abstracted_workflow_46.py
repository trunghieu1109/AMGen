async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Select Elements by Criteria Conformity]
    
    [Objective] 
    - Evaluate elements within a collection against defined criteria or conditions.
    - Identify elements that satisfy or fail to satisfy these criteria.
    
    [Agent Collaborations]
    - Debate and Self-Consistency Chain-of-Thought (SC_CoT)
    
    [Examples]
    - Subtask 1: Use SC_CoT to evaluate elements against criteria.
    - Subtask 2: Use Debate to finalize the selection of elements.
    """
    
    # Sub-task 1: Evaluate elements with SC_CoT
    cot_sc_instruction = "Sub-task 1: Evaluate elements against criteria using SC_CoT."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking1, answer1 = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, evaluating elements, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers.append(answer1.content)
        thinkingmapping[answer1.content] = thinking1
        answermapping[answer1.content] = answer1
    
    # Determine the most common answer
    most_common_answer = Counter(possible_answers).most_common(1)[0][0]
    thinking1 = thinkingmapping[most_common_answer]
    answer1 = answermapping[most_common_answer]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Evaluate and Prioritize Elements]
    
    [Objective] 
    - Evaluate a collection of elements against defined criteria.
    - Identify, select, and prioritize elements that best meet the specified conditions.
    
    [Agent Collaborations]
    - Debate and Chain-of-Thought (CoT)
    
    [Examples]
    - Subtask 1: Use CoT to evaluate and prioritize elements.
    - Subtask 2: Use Debate to finalize the prioritization.
    """
    
    # Sub-task 2: Prioritize elements with Debate
    debate_instruction_2 = "Sub-task 2: Prioritize elements using Debate."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            input_infos_2 = [taskInfo, thinking1, answer1]
            if r > 0:
                input_infos_2.extend(all_thinking2[r-1])
            thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, prioritizing elements, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + all_thinking2[-1] + all_answer2[-1], 
                                                 "Sub-task 2: Make final decision on prioritization.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, prioritizing elements, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    return final_answer
