async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Identify Constrained Combinations]
    
    [Objective] 
    - Enumerate or characterize all structured selections or arrangements from a finite set that satisfy specified constraints.
    - Identify, generate, or select subsets or combinations of elements from a given set that satisfy one or more specified constraints or relational conditions.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT) and Self-Consistency Chain-of-Thought (SC_CoT) are used to explore and validate possible combinations.
    
    [Examples]
    - Sub-task 1: Use CoT to enumerate possible combinations.
    - Sub-task 2: Use SC_CoT to validate and select the most consistent combinations.
    """
    
    # Sub-task 1: Identify constrained combinations using CoT
    cot_instruction = "Sub-task 1: Enumerate possible combinations that satisfy constraints from [taskInfo]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, enumerating combinations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Validate combinations using SC_CoT
    cot_sc_instruction = "Sub-task 2: Validate and select the most consistent combinations based on Sub-task 1 output"
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
    [Stage 2: Determine Scalar Transformation Characteristics]
    
    [Objective] 
    - Analyze a scalar mapping or relation to identify and characterize critical values or parameters that define its behavior.
    - Satisfy specified conditions or optimize given criteria.
    
    [Agent Collaborations]
    - Reflexion is used to iteratively refine the analysis of scalar transformations.
    
    [Examples]
    - Sub-task 3: Use Reflexion to analyze and refine scalar transformation characteristics.
    """
    
    # Sub-task 3: Determine scalar transformation characteristics using Reflexion
    cot_reflect_instruction = "Sub-task 3: Analyze scalar transformation characteristics based on Sub-task 1 and 2 outputs"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, analyzing scalar transformation, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 
                                       "Review scalar transformation analysis and provide feedback.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining scalar transformation, thinking: {thinking3.content}; answer: {answer3.content}")
    
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Final Decision Making]
    
    [Objective] 
    - Integrate outputs from all previous subtasks to generate the final answer for the query.
    
    [Agent Collaborations]
    - Debate is used to ensure comprehensive integration and validation of the final answer.
    
    [Examples]
    - Sub-task 4: Use Debate to integrate and validate the final answer.
    """
    
    # Sub-task 4: Generate final answer using Debate
    debate_instruction_5 = "Sub-task 4: Integrate outputs from all previous subtasks to generate the final answer"
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_5 = self.max_round
    
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3], 
                                           debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, integrating final answer, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                 "Sub-task 4: Make final decision on the final answer.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer