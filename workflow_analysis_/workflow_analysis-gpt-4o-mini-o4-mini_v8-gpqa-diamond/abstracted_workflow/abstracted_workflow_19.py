async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Stage 1: Select qualifying elements
    # Objective: Select elements that satisfy the defined requirements.
    # Agent Collaborations: Debate, CoT
    
    # Sub-task 1: Select elements using Debate and CoT
    debate_instruction = "Sub-task 1: Debate and select elements that satisfy the defined requirements."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking, answer = await agent([taskInfo], debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo] + all_thinking[r-1] + all_answer[r-1]
                thinking, answer = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting elements, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking, answer = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], 
                                                 "Sub-task 1: Make final decision on selected elements.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, selecting elements, thinking: {thinking.content}; answer: {answer.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking.content}; answer - {answer.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 2: Apply transformation
    # Objective: Apply the specified transformation to the selected elements. (optional)
    # Agent Collaborations: CoT, SC_CoT
    
    # Sub-task 2: Apply transformation using CoT and SC_CoT
    cot_instruction = "Sub-task 2: Apply the specified transformation to the selected elements."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent([taskInfo, thinking, answer], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, applying transformation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 3: Multi-criteria candidate selection
    # Objective: Identify elements that simultaneously satisfy multiple criteria.
    # Agent Collaborations: Debate, SC_CoT
    
    # Sub-task 3: Identify elements using Debate and SC_CoT
    debate_instruction_3 = "Sub-task 3: Debate and identify elements that satisfy multiple criteria."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_3 = self.max_round
    
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, identifying elements, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_3[r].append(thinking3)
            all_answer_3[r].append(answer3)
    
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking_3[-1] + all_answer_3[-1], 
                                                 "Sub-task 3: Make final decision on identified elements.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, identifying elements, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer
