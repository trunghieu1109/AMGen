async def forward_138(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze reaction conditions and structural features
    cot_instruction_0_1 = "Sub-task 1: Analyze the chemical reaction conditions (NaNO2, HCl, H2O) and understand the type of reaction they facilitate, particularly focusing on how they might lead to the formation of diketones."
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking0_1, answer0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing reaction conditions, thinking: {thinking0_1.content}; answer: {answer0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_sc_instruction_0_2 = "Sub-task 2: Characterize the structural features of the diketones produced: 4-isopropylcyclohexane-1,2-dione and 5-methylhexane-2,3-dione, to understand the necessary transformations from the starting materials."
    N_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N_0_2)]
    possible_answers_0_2 = []
    thinkingmapping_0_2 = {}
    answermapping_0_2 = {}
    
    for i in range(N_0_2):
        thinking0_2, answer0_2 = await cot_agents_0_2[i]([taskInfo, thinking0_1, answer0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, characterizing diketones, thinking: {thinking0_2.content}; answer: {answer0_2.content}")
        possible_answers_0_2.append(answer0_2.content)
        thinkingmapping_0_2[answer0_2.content] = thinking0_2
        answermapping_0_2[answer0_2.content] = answer0_2
    
    answer0_2 = Counter(possible_answers_0_2).most_common(1)[0][0]
    thinking0_2 = thinkingmapping_0_2[answer0_2]
    answer0_2 = answermapping_0_2[answer0_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking0_2.content}; answer - {answer0_2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 1: Evaluate starting materials
    cot_reflect_instruction_1_3 = "Sub-task 3: Evaluate the given choices of starting materials (A and B) to determine which structural features align with the diketones formation pathways identified in stage 0."
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    
    cot_inputs_1_3 = [taskInfo, thinking0_1, answer0_1, thinking0_2, answer0_2]
    
    thinking1_3, answer1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, evaluating starting materials, thinking: {thinking1_3.content}; answer: {answer1_3.content}")

    for i in range(N_max_1_3):
        feedback_1_3, correct_1_3 = await critic_agent_1_3([taskInfo, thinking1_3, answer1_3], 
                                       "please review the evaluation of starting materials and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, providing feedback, thinking: {feedback_1_3.content}; answer: {correct_1_3.content}")
        if correct_1_3.content == "True":
            break
        
        cot_inputs_1_3.extend([thinking1_3, answer1_3, feedback_1_3])
        
        thinking1_3, answer1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining evaluation of starting materials, thinking: {thinking1_3.content}; answer: {answer1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking1_3.content}; answer - {answer1_3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    cot_instruction_1_4 = "Sub-task 4: Select the starting materials that best match the required transformations to produce the specified diketones."
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1_4, answer1_4 = await cot_agent_1_4([taskInfo, thinking1_3, answer1_3], cot_instruction_1_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_4.id}, selecting starting materials, thinking: {thinking1_4.content}; answer: {answer1_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking1_4.content}; answer - {answer1_4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2: Verify and confirm starting materials
    debate_instruction_2_5 = "Sub-task 5: Apply the transformation conditions (NaNO2, HCl, H2O) to the selected starting materials to verify if they indeed produce the specified diketones."
    debate_agents_2_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_2_5 = self.max_round
    
    all_thinking_2_5 = [[] for _ in range(N_max_2_5)]
    all_answer_2_5 = [[] for _ in range(N_max_2_5)]
    
    for r in range(N_max_2_5):
        for i, agent in enumerate(debate_agents_2_5):
            if r == 0:
                thinking2_5, answer2_5 = await agent([taskInfo, thinking1_4, answer1_4], 
                                           debate_instruction_2_5, r, is_sub_task=True)
            else:
                input_infos_2_5 = [taskInfo, thinking1_4, answer1_4] + all_thinking_2_5[r-1] + all_answer_2_5[r-1]
                thinking2_5, answer2_5 = await agent(input_infos_2_5, debate_instruction_2_5, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, verifying starting materials, thinking: {thinking2_5.content}; answer: {answer2_5.content}")
            all_thinking_2_5[r].append(thinking2_5)
            all_answer_2_5[r].append(answer2_5)
    
    final_decision_agent_2_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking2_5, answer2_5 = await final_decision_agent_2_5([taskInfo] + all_thinking_2_5[-1] + all_answer_2_5[-1], 
                                                 "Sub-task 5: Make final decision on verification of starting materials.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, verifying starting materials, thinking: {thinking2_5.content}; answer: {answer2_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking2_5.content}; answer - {answer2_5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    cot_instruction_2_6 = "Sub-task 6: Confirm the correctness of the selected starting materials by comparing the reaction outcomes with the expected diketones."
    cot_agent_2_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2_6, answer2_6 = await cot_agent_2_6([taskInfo, thinking2_5, answer2_5], cot_instruction_2_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_6.id}, confirming starting materials, thinking: {thinking2_6.content}; answer: {answer2_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking2_6.content}; answer - {answer2_6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_6, answer2_6, sub_tasks, agents)
    return final_answer