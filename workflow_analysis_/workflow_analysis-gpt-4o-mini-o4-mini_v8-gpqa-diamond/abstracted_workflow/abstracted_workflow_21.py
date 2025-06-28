async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Stage 1: Apply Transformation
    # Objective: Apply a specified transformation or operation to input entities to produce corresponding output representations.
    # Agent Collaborations: CoT, SC_CoT
    
    # Sub-task 1: Apply transformation using CoT
    cot_instruction = "Sub-task 1: Apply the specified transformation to the input entities with context from [taskInfo]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, applying transformation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Assess transformation impact using SC_CoT
    cot_sc_instruction = "Sub-task 2: Assess the impact of the transformation on the input entities"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, assessing impact, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Assess Modification Impact
    # Objective: Assess the effect of a specified change on the input or system by determining the resulting state or outcome and characterizing its impact.
    # Agent Collaborations: SC_CoT, Reflexion
    
    # Sub-task 3: Assess modification impact using Reflexion
    cot_reflect_instruction = "Sub-task 3: Assess the modification impact based on previous outputs"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, assessing modification impact, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 
                                       "Review the modification impact assessment and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining modification impact assessment, thinking: {thinking3.content}; answer: {answer3.content}")
    
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 3: Generate Variant Configurations
    # Objective: Systematically produce alternative configurations or representations by applying a defined transformation under specified constraints.
    # Agent Collaborations: Debate, CoT
    
    # Sub-task 4: Generate variant configurations using Debate
    debate_instruction = "Sub-task 4: Generate variant configurations based on previous outputs"
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking3, answer3], 
                                           debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3, answer3] + all_thinking[r-1] + all_answer[r-1]
                thinking, answer = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, generating variant configurations, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking, answer = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], 
                                                 "Sub-task 4: Make final decision on variant configurations.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, deciding on variant configurations, thinking: {thinking.content}; answer: {answer.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking.content}; answer - {answer.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 4: Evaluate and Select Candidates
    # Objective: Evaluate a set of elements against defined criteria and select those that conform to the requirements.
    # Agent Collaborations: Debate, SC_CoT
    
    # Sub-task 5: Evaluate and select candidates using Debate and SC_CoT
    debate_instruction_5 = "Sub-task 5: Evaluate and select candidates based on previous outputs"
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_5 = self.max_round
    
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking, answer], 
                                           debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking, answer] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating and selecting candidates, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                 "Sub-task 5: Make final decision on selected candidates.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, selecting candidates, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer