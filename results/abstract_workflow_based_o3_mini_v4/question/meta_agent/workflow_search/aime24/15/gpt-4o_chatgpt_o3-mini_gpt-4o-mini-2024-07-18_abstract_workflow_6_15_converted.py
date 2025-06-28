async def forward_15(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []

    # Stage 1
    # Sub-task 1
    cot_instruction_1 = "Sub-task 1: Calculate the total number of residents owning at least one of the non-candy items (diamond ring, golf clubs, garden spade) using basic overlapping scenarios."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculating residents with at least one non-candy item, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2
    cot_sc_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, identify the number of residents who own exactly one of the three non-candy items."
    N_2 = self.max_sc
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    
    for i, agent in enumerate(cot_sc_agents_2):
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, determining residents owning exactly one non-candy item, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    most_common_answerstr_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answerstr_2]
    answer2 = answermapping_2[most_common_answerstr_2]
    sub_tasks.append(f"Sub-task 2 output (most common): thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3
    cot_instruction_3 = "Sub-task 3: Identify the number of residents who own exactly two of the three non-candy items."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, finding residents with exactly two non-candy items, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4
    cot_instruction_4 = "Sub-task 4: Identify the total number of residents who own exactly three of these non-candy items."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, finding residents owning exactly three non-candy items, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2
    # Sub-task 5
    debate_instruction_5 = "Sub-task 5: Compute the total number of residents who own at least one of the three non-candy items using inclusion-exclusion principle."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            input_infos_5 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
            if r > 0:
                input_infos_5.extend(all_thinking5[r-1])
            
            thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating residents owning at least one non-candy item with IE principle, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
        
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on residents owning at least one non-candy item.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining residents owning at least one non-candy item, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6
    cot_instruction_6 = "Sub-task 6: Determine the number of residents who own all three non-candy items (diamond ring, golf clubs, and garden spade)."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking3, answer3, thinking4, answer4], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, determining residents owning all non-candy items, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Stage 3
    # Sub-task 7
    reflect_instruction_7 = "Sub-task 7: Use the results to find the number of residents who own all four items."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking5, answer5, thinking6, answer6]
    
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, finding residents owning all items, thinking: {thinking7.content}; answer: {answer7.content}")

    for i in range(N_max_7):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7], "Please review the residents owning all items calculation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining residents owning all items, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    # Stage 4
    # Sub-task 8
    reflect_instruction_8 = "Sub-task 8: Calculate the final answer by confirming the number of residents who own all items satisfies the total population and conditions provided."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_8 = self.max_round
    cot_inputs_8 = [taskInfo, thinking7, answer7]
    
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, reflect_instruction_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, calculating final verification of residents owning all items, thinking: {thinking8.content}; answer: {answer8.content}")

    for i in range(N_max_8):
        feedback, correct = await critic_agent_8([taskInfo, thinking8, answer8], "Please verify the final adjustment matches all conditions.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_8.extend([thinking8, answer8, feedback])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, reflect_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining the final answer verification, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Subtask 8 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer
