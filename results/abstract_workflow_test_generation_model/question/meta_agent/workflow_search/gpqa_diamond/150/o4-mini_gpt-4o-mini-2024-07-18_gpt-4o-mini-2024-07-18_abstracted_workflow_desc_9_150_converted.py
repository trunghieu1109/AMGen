async def forward_150(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Extract the state vector ψ from the query'
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting state vector, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_instruction = 'Sub-task 2: Extract the observable operator P as a 3×3 numeric matrix from the query'
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking2, answer2 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting operator P, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction = 'Sub-task 3: Extract the list of answer choices and their numeric values from the query'
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting answer choices, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_sc_instruction = 'Sub-task 4: Compute all eigenvalues and eigenvectors of the observable matrix P'
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking4_i, answer4_i = await cot_agents[i]([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing eigenpairs, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_answers.append(answer4_i.content)
        thinkingmapping[answer4_i.content] = thinking4_i
        answermapping[answer4_i.content] = answer4_i
    answer4_content = Counter(possible_answers).most_common(1)[0][0]
    thinking4 = thinkingmapping[answer4_content]
    answer4 = answermapping[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction = 'Sub-task 5: Identify which eigenvalue equals 0 and collect its corresponding eigenvector(s)'
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_instruction, "context": ["user query", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "CoT"}
    thinking5, answer5 = await cot_agent([taskInfo, thinking4, answer4], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying zero eigenvalue eigenspace, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_reflect_instruction = 'Sub-task 6: Construct the projector operator onto the eigenspace associated with eigenvalue 0'
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking5, answer5]
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_reflect_instruction, "context": ["user query", "thinking of subtask 5", "answer of subtask 5"], "agent_collaboration": "Reflexion"}
    thinking6, answer6 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, initial projector construction, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking6, answer6], "Please review the projector construction and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refined projector, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    cot_instruction = 'Sub-task 7: Apply the projector to the state vector ψ to obtain the projected component ψ_proj'
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 6", "answer of subtask 6"], "agent_collaboration": "CoT"}
    thinking7, answer7 = await cot_agent([taskInfo, thinking1, answer1, thinking6, answer6], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, applying projector, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    cot_sc_instruction = 'Sub-task 8: Compute the probability of measuring 0 by calculating the squared norm of ψ_proj'
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 7", "answer of subtask 7"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking8_i, answer8_i = await cot_agents[i]([taskInfo, thinking7, answer7], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing probability, thinking: {thinking8_i.content}; answer: {answer8_i.content}")
        possible_answers.append(answer8_i.content)
        thinkingmapping[answer8_i.content] = thinking8_i
        answermapping[answer8_i.content] = answer8_i
    answer8_content = Counter(possible_answers).most_common(1)[0][0]
    thinking8 = thinkingmapping[answer8_content]
    answer8 = answermapping[answer8_content]
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    debate_instruction = 'Sub-task 9: Compare the computed probability with the extracted answer choices and select the matching letter'
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    subtask_desc9 = {"subtask_id": "subtask_9", "instruction": debate_instruction, "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 8", "answer of subtask 8"], "agent_collaboration": "Debate"}
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking9_i, answer9_i = await agent([taskInfo, thinking3, answer3, thinking8, answer8], debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3, answer3, thinking8, answer8] + all_thinking[r-1] + all_answer[r-1]
                thinking9_i, answer9_i = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing probability and choices, thinking: {thinking9_i.content}; answer: {answer9_i.content}")
            all_thinking[r].append(thinking9_i)
            all_answer[r].append(answer9_i)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], "Sub-task 9: Make final decision on the correct multiple-choice letter.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting answer choice, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs