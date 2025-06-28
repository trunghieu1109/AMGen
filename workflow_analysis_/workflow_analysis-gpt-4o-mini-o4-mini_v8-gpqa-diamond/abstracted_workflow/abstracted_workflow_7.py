async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Stage 1: Select Qualifying Elements
    # Objective: Select elements that satisfy the specified requirements.
    # Agent Collaborations: Debate, CoT
    
    # Sub-task 1: Select elements using Debate
    debate_instruction = "Sub-task 1: Debate to select elements that satisfy the specified requirements."
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

    # Stage 2: Evaluate and Characterize Input Attributes
    # Objective: Analyze and interpret defining attributes, relationships, or behaviors of inputs to characterize them.
    # Agent Collaborations: SC_CoT, CoT
    
    # Sub-task 2: Evaluate attributes using SC_CoT
    cot_sc_instruction = "Sub-task 2: Evaluate and characterize input attributes based on selected elements."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking, answer = await cot_agents[i]([taskInfo, thinking, answer], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, evaluating attributes, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers.append(answer.content)
        thinkingmapping[answer.content] = thinking
        answermapping[answer.content] = answer
    
    answer = Counter(possible_answers).most_common(1)[0][0]
    thinking = thinkingmapping[answer]
    answer = answermapping[answer]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking.content}; answer - {answer.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 3: Derive Quantitative Measure
    # Objective: Derive a quantitative measure by applying defined rules or procedures to input data.
    # Agent Collaborations: CoT, SC_CoT
    
    # Sub-task 3: Derive measure using Reflexion
    cot_reflect_instruction = "Sub-task 3: Derive quantitative measure based on characterized attributes."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking, answer]
    
    thinking, answer = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, deriving measure, thinking: {thinking.content}; answer: {answer.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking, answer], 
                                       "Review derived measure for accuracy and provide limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs.extend([thinking, answer, feedback])
        thinking, answer = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining measure, thinking: {thinking.content}; answer: {answer.content}")
    
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking.content}; answer - {answer.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking, answer, sub_tasks, agents)
    return final_answer
}