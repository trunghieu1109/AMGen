async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs =  []
    
    cot_instruction_1 = "Sub-task 1: Identify the condition under which n^4 + 1 is divisible by p^2 for a prime p."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing condition, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Enumerate candidate primes p that are congruent to 1 modulo 8."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of stage_1_subtask_1", "answer of stage_1_subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining candidate primes, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_3 = "Sub-task 3: Test whether the congruence x^4 ≡ -1 (mod p) has a solution for each candidate prime."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "stage_1_subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "answer of stage_1_subtask_2"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, testing congruence, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Select the smallest prime p for which the congruence x^4 ≡ -1 (mod p) has a solution."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "stage_1_subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "answer of stage_1_subtask_3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, selecting smallest prime, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_5 = "Sub-task 5: Verify that the solution modulo p can be lifted to a solution modulo p^2."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": cot_instruction_5,
        "context": ["user query", "answer of stage_1_subtask_4"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, verifying lifting, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_sc_instruction_6 = "Sub-task 6: Compute the least positive integer m such that m^4 + 1 ≡ 0 (mod p^2)."
    N = self.max_sc
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6 = []
    thinkingmapping_6 = {}
    answermapping_6 = {}
    subtask_desc6 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", "answer of stage_2_subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, computing least m, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6.content)
        thinkingmapping_6[answer6.content] = thinking6
        answermapping_6[answer6.content] = answer6
    answer6_content = Counter(possible_answers_6).most_common(1)[0][0]
    thinking6 = thinkingmapping_6[answer6_content]
    answer6 = answermapping_6[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_reflect_instruction_7 = "Sub-task 7: Embed verification steps to ensure that each subtask's outcome is correct."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6, answer6]
    subtask_desc7 = {
        "subtask_id": "stage_3_subtask_1",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", "thinking of stage_2_subtask_2", "answer of stage_2_subtask_2"],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, verifying outcomes, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7], 
                                       "please review the verification steps and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining verification, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_reflect_instruction_8 = "Sub-task 8: Reflect on the entire process to ensure no errors have propagated."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max_8 = self.max_round
    cot_inputs_8 = [taskInfo, thinking7, answer7]
    subtask_desc8 = {
        "subtask_id": "stage_3_subtask_2",
        "instruction": cot_reflect_instruction_8,
        "context": ["user query", "thinking of stage_3_subtask_1", "answer of stage_3_subtask_1"],
        "agent_collaboration": "Reflexion"
    }
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, reflecting on process, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(N_max_8):
        feedback, correct = await critic_agent_8([taskInfo, thinking8, answer8], 
                                       "please review the reflection process and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_8.extend([thinking8, answer8, feedback])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining reflection, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {
        "thinking": thinking8,
        "answer": answer8
    }
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs