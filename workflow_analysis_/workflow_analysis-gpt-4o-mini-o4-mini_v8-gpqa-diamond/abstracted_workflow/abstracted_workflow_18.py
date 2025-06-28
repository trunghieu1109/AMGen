async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Stage 1: Extract Defining Features
    # Objective: Isolate and characterize essential components, attributes, and relationships of the input.
    # Agent Collaborations: CoT, Debate
    
    # Sub-task 1: Extract features using CoT
    cot_instruction = "Sub-task 1: Extract defining features from the input, focusing on essential components and relationships."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting features, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Debate on feature extraction
    debate_instruction = "Sub-task 2: Debate on the extracted features to ensure comprehensive coverage and accuracy."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking1, answer1], 
                                               debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking1, answer1] + all_thinking[r-1] + all_answer[r-1]
                thinking, answer = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, debating features, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], 
                                                    "Sub-task 2: Make final decision on extracted features.", 
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, deciding on features, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Classify Inputs Based on Attributes and Relationships
    # Objective: Assign input entities to categories based on their attributes and relationships.
    # Agent Collaborations: CoT, SC_CoT
    
    # Sub-task 3: Classify inputs using CoT
    cot_instruction = "Sub-task 3: Classify inputs based on extracted features and relationships."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent([taskInfo, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, classifying inputs, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Self-consistency classification
    cot_sc_instruction = "Sub-task 4: Refine classification through self-consistency checks."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking4, answer4 = await cot_agents[i]([taskInfo, thinking3, answer3], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, refining classification, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers.append(answer4.content)
        thinkingmapping[answer4.content] = thinking4
        answermapping[answer4.content] = answer4
    
    answer4 = Counter(possible_answers).most_common(1)[0][0]
    thinking4 = thinkingmapping[answer4]
    answer4 = answermapping[answer4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 3: Derive Quantitative Measure
    # Objective: Compute a meaningful numeric metric from the input data.
    # Agent Collaborations: CoT, SC_CoT
    
    # Sub-task 5: Derive measure using CoT
    cot_instruction = "Sub-task 5: Derive a quantitative measure from classified inputs."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent([taskInfo, thinking4, answer4], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, deriving measure, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Self-consistency measure derivation
    cot_sc_instruction = "Sub-task 6: Refine measure derivation through self-consistency checks."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking6, answer6 = await cot_agents[i]([taskInfo, thinking5, answer5], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, refining measure, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers.append(answer6.content)
        thinkingmapping[answer6.content] = thinking6
        answermapping[answer6.content] = answer6
    
    answer6 = Counter(possible_answers).most_common(1)[0][0]
    thinking6 = thinkingmapping[answer6]
    answer6 = answermapping[answer6]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Stage 4: Apply Transformation
    # Objective: Transform input values using the specified operation.
    # Agent Collaborations: SC_CoT, Reflexion
    
    # Sub-task 7: Apply transformation using SC_CoT
    cot_sc_instruction = "Sub-task 7: Apply transformation to derived measure using self-consistency checks."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking7, answer7 = await cot_agents[i]([taskInfo, thinking6, answer6], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, applying transformation, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers.append(answer7.content)
        thinkingmapping[answer7.content] = thinking7
        answermapping[answer7.content] = answer7
    
    answer7 = Counter(possible_answers).most_common(1)[0][0]
    thinking7 = thinkingmapping[answer7]
    answer7 = answermapping[answer7]
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    
    print("Subtask 7 answer: ", sub_tasks[-1])

    # Sub-task 8: Reflexion on transformation
    cot_reflect_instruction = "Sub-task 8: Reflect on the transformation process to ensure accuracy and completeness."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking7, answer7]
    
    thinking8, answer8 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, reflecting on transformation, thinking: {thinking8.content}; answer: {answer8.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking8, answer8], 
                                       "Review transformation for accuracy and provide feedback.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs.extend([thinking8, answer8, feedback])
        thinking8, answer8 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining transformation, thinking: {thinking8.content}; answer: {answer8.content}")
    
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    
    print("Subtask 8 answer: ", sub_tasks[-1])

    # Stage 5: Select Qualifying Elements
    # Objective: Select elements that satisfy the defined criteria.
    # Agent Collaborations: Debate, CoT
    
    # Sub-task 9: Select elements using Debate
    debate_instruction = "Sub-task 9: Debate on the selection of elements that meet the criteria."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max = self.max_round
    
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking9, answer9 = await agent([taskInfo, thinking8, answer8], 
                                               debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking8, answer8] + all_thinking[r-1] + all_answer[r-1]
                thinking9, answer9 = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, selecting elements, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking[r].append(thinking9)
            all_answer[r].append(answer9)
    
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], 
                                                    "Sub-task 9: Make final decision on selected elements.", 
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, deciding on elements, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    
    print("Subtask 9 answer: ", sub_tasks[-1])

    # Stage 6: Integrate Inputs
    # Objective: Combine multiple input elements into a single consolidated output.
    # Agent Collaborations: SC_CoT, CoT
    
    # Sub-task 10: Integrate inputs using SC_CoT
    cot_sc_instruction = "Sub-task 10: Integrate selected elements into a consolidated output using self-consistency checks."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking10, answer10 = await cot_agents[i]([taskInfo, thinking9, answer9], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, integrating inputs, thinking: {thinking10.content}; answer: {answer10.content}")
        possible_answers.append(answer10.content)
        thinkingmapping[answer10.content] = thinking10
        answermapping[answer10.content] = answer10
    
    answer10 = Counter(possible_answers).most_common(1)[0][0]
    thinking10 = thinkingmapping[answer10]
    answer10 = answermapping[answer10]
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    
    print("Subtask 10 answer: ", sub_tasks[-1])

    # Sub-task 11: Final integration using CoT
    cot_instruction = "Sub-task 11: Finalize the integration of inputs into a single output."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking11, answer11 = await cot_agent([taskInfo, thinking10, answer10], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, finalizing integration, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    
    print("Subtask 11 answer: ", sub_tasks[-1])

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking11, answer11, sub_tasks, agents)
    return final_answer