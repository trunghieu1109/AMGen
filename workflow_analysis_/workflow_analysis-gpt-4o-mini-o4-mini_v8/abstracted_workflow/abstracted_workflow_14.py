async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Stage 1: Derive Primary Variable
    # Objective: Compute quantitative outputs by applying transformations to input values.
    # Agent Collaborations: CoT, SC_CoT
    cot_instruction = "Sub-task 1: Derive primary variable by applying transformations to input values."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, deriving primary variable, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    # Stage 2: Validate Transformation Output
    # Objective: Verify that the output satisfies conditions without violation.
    # Agent Collaborations: CoT, Reflexion
    cot_reflect_instruction = "Sub-task 2: Validate transformation output against defined conditions."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1]
    thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, validating transformation output, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking2, answer2], 
                                       "Review validation of transformation output and provide feedback.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining validation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    # Stage 3: Compute Adjusted Aggregate Measure
    # Objective: Transform and aggregate input elements to produce a summary output.
    # Agent Collaborations: CoT, Reflexion
    cot_reflect_instruction = "Sub-task 3: Compute adjusted aggregate measure from input elements."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, computing adjusted aggregate measure, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 
                                       "Review computation of adjusted aggregate measure and provide feedback.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining computation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    # Stage 4: Simplify Ratio and Aggregate Components
    # Objective: Simplify quantitative relationships and compute aggregate values.
    # Agent Collaborations: Debate, Reflexion
    debate_instruction = "Sub-task 4: Simplify ratio and aggregate components to compute final output."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], 
                                           debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3, answer3] + all_thinking[r-1] + all_answer[r-1]
                thinking4, answer4 = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying ratio and computing final output, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking[r].append(thinking4)
            all_answer[r].append(answer4)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], 
                                                 "Sub-task 4: Make final decision on final output.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final output, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer