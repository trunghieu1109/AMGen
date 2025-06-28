async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Determine Threshold Parameter]
    
    [Objective] 
    - Identify and quantify the threshold parameter that governs subset configurations.
    
    [Agent Collaborations]
    - Use Chain-of-Thought (CoT) to consider all possible scenarios.
    - Use Self-Consistency Chain-of-Thought (SC_CoT) to build consensus on the threshold parameter.
    
    [Examples]
    - Sub-task 1: Analyze the problem to identify potential threshold parameters using CoT.
    - Sub-task 2: Use SC_CoT to reach a consensus on the most likely threshold parameter.
    """
    
    # Sub-task 1: Determine potential threshold parameters using CoT
    cot_instruction = "Sub-task 1: Consider/calculate all possible cases of threshold parameters with context from taskInfo."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, consider/calculate all possible scenarios of threshold parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Reach consensus on threshold parameter using SC_CoT
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, consider/calculate potential cases of threshold parameters."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, consider all possible cases of threshold parameters, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Simplify Ratio and Aggregate Components]
    
    [Objective] 
    - Transform the quantitative relationship into simplest integer-based components and compute the aggregate value.
    
    [Agent Collaborations]
    - Use Debate to explore different simplification strategies.
    - Use Reflexion to refine and validate the final aggregate value.
    
    [Examples]
    - Sub-task 3: Debate different strategies for simplifying the ratio.
    - Sub-task 4: Use Reflexion to validate and refine the aggregate value.
    """
    
    # Sub-task 3: Debate strategies for simplifying the ratio
    debate_instruction = "Sub-task 3: Debate different strategies for simplifying the ratio and calculating the aggregate value."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2, answer2], 
                                           debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking2, answer2] + all_thinking[r-1] + all_answer[r-1]
                thinking, answer = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying ratio and calculating aggregate value, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    
    # Sub-task 4: Validate and refine the aggregate value using Reflexion
    cot_reflect_instruction = "Sub-task 4: Validate and refine the aggregate value using Reflexion."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    
    cot_inputs = [taskInfo, thinking2, answer2] + all_thinking[-1] + all_answer[-1]
    
    thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, validating aggregate value, thinking: {thinking4.content}; answer: {answer4.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking4, answer4], 
                                       "Review the aggregate value for completeness and accuracy.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining aggregate value, thinking: {thinking4.content}; answer: {answer4.content}")
    
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
