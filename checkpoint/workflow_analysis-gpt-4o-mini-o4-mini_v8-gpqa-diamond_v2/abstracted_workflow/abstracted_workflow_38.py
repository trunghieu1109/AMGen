async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Evaluate and prioritize elements]
    
    [Objective] 
    - Evaluate a collection of elements against defined criteria to identify, select, and prioritize those that satisfy or best meet the specified conditions.
    
    [Agent Collaborations]
    - Use Debate and Chain-of-Thought (CoT) patterns to evaluate and prioritize elements.
    
    [Examples]
    - Implement subtasks to evaluate elements using Debate and CoT.
    """
    
    # Sub-task 1: Evaluate elements using Debate
    debate_instruction_1 = "Sub-task 1: Evaluate elements based on criteria and prioritize them."
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking1 = [[] for _ in range(N_max_1)]
    all_answer1 = [[] for _ in range(N_max_1)]
    
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            input_infos_1 = [taskInfo]
            thinking1, answer1 = await agent(input_infos_1, debate_instruction_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating elements, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking1[r].append(thinking1)
            all_answer1[r].append(answer1)
    
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + all_thinking1[-1] + all_answer1[-1], "Sub-task 1: Make final decision on prioritized elements.", is_sub_task=True)
    agents.append(f"Final Decision agent on prioritizing elements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Apply transformation]
    
    [Objective] 
    - Apply a specified operation or transformation to an input to produce a corresponding output.
    
    [Agent Collaborations]
    - Use Chain-of-Thought (CoT) and Self-Consistency Chain-of-Thought (SC_CoT) patterns to apply transformations.
    
    [Examples]
    - Implement subtasks to apply transformations using CoT and SC_CoT.
    """
    
    # Sub-task 2: Apply transformation using CoT
    cot_instruction_2 = "Sub-task 2: Apply transformation to the prioritized elements."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, applying transformation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Multi-criteria selection]
    
    [Objective] 
    - Identify or select element(s) from a set that simultaneously satisfy multiple defined criteria or conditions.
    
    [Agent Collaborations]
    - Use Debate and Self-Consistency Chain-of-Thought (SC_CoT) patterns for multi-criteria selection.
    
    [Examples]
    - Implement subtasks for multi-criteria selection using Debate and SC_CoT.
    """
    
    # Sub-task 3: Multi-criteria selection using SC_CoT
    cot_sc_instruction_3 = "Sub-task 3: Select elements based on multiple criteria."
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, selecting elements, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    
    most_common_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[most_common_answer_3]
    answer3 = answermapping_3[most_common_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer