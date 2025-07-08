async def forward_154(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Write down the explicit form of the given state vector |\u03C8\u27E9 in the eigenbasis of P_z, as provided: (-1/2, 1/\u221A2, -1/2)."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, write explicit state vector, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    cot_instruction_2 = "Sub-task 2: Write down the matrix form of the operator P_z as given, and identify its eigenvalues corresponding to the basis vectors."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, write matrix P_z and eigenvalues, thinking: {thinking2.content}; answer: {answer2.content}")
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
    cot_instruction_3 = "Sub-task 3: Calculate the expectation value \u27E8P_z\u27E9 = \u27E8\u03C8|P_z|\u03C8\u27E9 by performing the matrix multiplication explicitly, showing all intermediate steps."
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, calculate expectation \u27E8P_z\u27E9, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    cot_instruction_4_1 = "Sub-task 4.1: Compute the square of each eigenvalue of P_z explicitly."
    cot_instruction_4_2 = "Sub-task 4.2: Compute the squared modulus of each component of |\u03C8\u27E9 explicitly."
    cot_instruction_4_3 = "Sub-task 4.3: Multiply each squared eigenvalue by the corresponding squared component explicitly."
    cot_instruction_4_4 = "Sub-task 4.4: Sum all these products to find \u27E8P_z^2\u27E9, showing all intermediate calculations explicitly to avoid arithmetic errors."
    cot_agents_4_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_4_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_4_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_4_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4_1 = []
    thinkingmapping_4_1 = {}
    answermapping_4_1 = {}
    subtask_desc4_1 = {
        "subtask_id": "subtask_4_1",
        "instruction": cot_instruction_4_1,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4_1, answer4_1 = await cot_agents_4_1[i]([taskInfo, thinking2, answer2], cot_instruction_4_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4_1[i].id}, compute squared eigenvalues, thinking: {thinking4_1.content}; answer: {answer4_1.content}")
        possible_answers_4_1.append(answer4_1.content)
        thinkingmapping_4_1[answer4_1.content] = thinking4_1
        answermapping_4_1[answer4_1.content] = answer4_1
    answer4_1_content = Counter(possible_answers_4_1).most_common(1)[0][0]
    thinking4_1 = thinkingmapping_4_1[answer4_1_content]
    answer4_1 = answermapping_4_1[answer4_1_content]
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking4_1.content}; answer - {answer4_1.content}")
    subtask_desc4_1['response'] = {
        "thinking": thinking4_1,
        "answer": answer4_1
    }
    logs.append(subtask_desc4_1)
    possible_answers_4_2 = []
    thinkingmapping_4_2 = {}
    answermapping_4_2 = {}
    subtask_desc4_2 = {
        "subtask_id": "subtask_4_2",
        "instruction": cot_instruction_4_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4_2, answer4_2 = await cot_agents_4_2[i]([taskInfo, thinking1, answer1], cot_instruction_4_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4_2[i].id}, compute squared components of state vector, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
        possible_answers_4_2.append(answer4_2.content)
        thinkingmapping_4_2[answer4_2.content] = thinking4_2
        answermapping_4_2[answer4_2.content] = answer4_2
    answer4_2_content = Counter(possible_answers_4_2).most_common(1)[0][0]
    thinking4_2 = thinkingmapping_4_2[answer4_2_content]
    answer4_2 = answermapping_4_2[answer4_2_content]
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking4_2.content}; answer - {answer4_2.content}")
    subtask_desc4_2['response'] = {
        "thinking": thinking4_2,
        "answer": answer4_2
    }
    logs.append(subtask_desc4_2)
    possible_answers_4_3 = []
    thinkingmapping_4_3 = {}
    answermapping_4_3 = {}
    subtask_desc4_3 = {
        "subtask_id": "subtask_4_3",
        "instruction": cot_instruction_4_3,
        "context": ["user query", "thinking of subtask 4_1", "answer of subtask 4_1", "thinking of subtask 4_2", "answer of subtask 4_2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4_3, answer4_3 = await cot_agents_4_3[i]([taskInfo, thinking4_1, answer4_1, thinking4_2, answer4_2], cot_instruction_4_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4_3[i].id}, multiply squared eigenvalues by squared components, thinking: {thinking4_3.content}; answer: {answer4_3.content}")
        possible_answers_4_3.append(answer4_3.content)
        thinkingmapping_4_3[answer4_3.content] = thinking4_3
        answermapping_4_3[answer4_3.content] = answer4_3
    answer4_3_content = Counter(possible_answers_4_3).most_common(1)[0][0]
    thinking4_3 = thinkingmapping_4_3[answer4_3_content]
    answer4_3 = answermapping_4_3[answer4_3_content]
    sub_tasks.append(f"Sub-task 4.3 output: thinking - {thinking4_3.content}; answer - {answer4_3.content}")
    subtask_desc4_3['response'] = {
        "thinking": thinking4_3,
        "answer": answer4_3
    }
    logs.append(subtask_desc4_3)
    possible_answers_4_4 = []
    thinkingmapping_4_4 = {}
    answermapping_4_4 = {}
    subtask_desc4_4 = {
        "subtask_id": "subtask_4_4",
        "instruction": cot_instruction_4_4,
        "context": ["user query", "thinking of subtask 4_3", "answer of subtask 4_3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4_4, answer4_4 = await cot_agents_4_4[i]([taskInfo, thinking4_3, answer4_3], cot_instruction_4_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4_4[i].id}, sum products to find expectation \u27E8P_z^2\u27E9, thinking: {thinking4_4.content}; answer: {answer4_4.content}")
        possible_answers_4_4.append(answer4_4.content)
        thinkingmapping_4_4[answer4_4.content] = thinking4_4
        answermapping_4_4[answer4_4.content] = answer4_4
    answer4_4_content = Counter(possible_answers_4_4).most_common(1)[0][0]
    thinking4_4 = thinkingmapping_4_4[answer4_4_content]
    answer4_4 = answermapping_4_4[answer4_4_content]
    sub_tasks.append(f"Sub-task 4.4 output: thinking - {thinking4_4.content}; answer - {answer4_4.content}")
    subtask_desc4_4['response'] = {
        "thinking": thinking4_4,
        "answer": answer4_4
    }
    logs.append(subtask_desc4_4)
    cot_instruction_5 = "Sub-task 5: Verify the calculations of \u27E8P_z\u27E9 and \u27E8P_z^2\u27E9 independently by re-computing and cross-checking each term to ensure no arithmetic mistakes."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_5 = [taskInfo, thinking3, answer3, thinking4_4, answer4_4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4_4", "answer of subtask 4_4"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, verify expectation values, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "please review the verification of expectation values and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining verification, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    cot_instruction_6 = "Sub-task 6: Compute the uncertainty \u0394P_z using the formula \u0394P_z = sqrt(\u27E8P_z^2\u27E9 - \u27E8P_z\u27E9^2) with the verified expectation values."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, compute uncertainty \u0394P_z, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "please review the uncertainty calculation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining uncertainty \u0394P_z, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    debate_instruction_7 = "Sub-task 7: Compare the computed uncertainty \u0394P_z with the given multiple-choice options and select the correct answer choice (A, B, C, or D)."
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, select correct choice, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the correct uncertainty \u0394P_z choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs