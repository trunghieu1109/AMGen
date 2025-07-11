async def forward_188(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Define spontaneous symmetry breaking and the criterion for an effective particle to be associated with a spontaneously broken continuous symmetry."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, defining SSB, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    cot_instruction2 = "Sub-task 2: Extract the list of effective particles mentioned in the query: Magnon, Skyrmion, Pion, Phonon."
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_instruction2, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking2, answer2 = await cot_agent2([taskInfo], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, extracting particle list, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    cot_sc_instruction = "Identify which continuous symmetry is spontaneously broken to give rise to {}."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc3_1 = {"subtask_id": "subtask_3.1", "instruction": cot_sc_instruction.format("magnons"), "context": ["user query", "response of subtask_1", "response of subtask_2"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking_i, answer_i = await cot_agents[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction.format("magnons"), is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, identify broken symmetry for magnons, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers.append(answer_i.content)
        thinkingmapping[answer_i.content] = thinking_i
        answermapping[answer_i.content] = answer_i
    answer31_content = Counter(possible_answers).most_common(1)[0][0]
    thinking3_1 = thinkingmapping[answer31_content]
    answer3_1 = answermapping[answer31_content]
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    print("Step 3.1: ", sub_tasks[-1])
    subtask_desc3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc3_1)
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc3_2 = {"subtask_id": "subtask_3.2", "instruction": cot_sc_instruction.format("skyrmions"), "context": ["user query", "response of subtask_1", "response of subtask_2"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking_i, answer_i = await cot_agents[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction.format("skyrmions"), is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, identify broken symmetry for skyrmions, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers.append(answer_i.content)
        thinkingmapping[answer_i.content] = thinking_i
        answermapping[answer_i.content] = answer_i
    answer32_content = Counter(possible_answers).most_common(1)[0][0]
    thinking3_2 = thinkingmapping[answer32_content]
    answer3_2 = answermapping[answer32_content]
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    print("Step 3.2: ", sub_tasks[-1])
    subtask_desc3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc3_2)
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc3_3 = {"subtask_id": "subtask_3.3", "instruction": cot_sc_instruction.format("pions"), "context": ["user query", "response of subtask_1", "response of subtask_2"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking_i, answer_i = await cot_agents[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction.format("pions"), is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, identify broken symmetry for pions, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers.append(answer_i.content)
        thinkingmapping[answer_i.content] = thinking_i
        answermapping[answer_i.content] = answer_i
    answer33_content = Counter(possible_answers).most_common(1)[0][0]
    thinking3_3 = thinkingmapping[answer33_content]
    answer3_3 = answermapping[answer33_content]
    sub_tasks.append(f"Sub-task 3.3 output: thinking - {thinking3_3.content}; answer - {answer3_3.content}")
    print("Step 3.3: ", sub_tasks[-1])
    subtask_desc3_3['response'] = {"thinking": thinking3_3, "answer": answer3_3}
    logs.append(subtask_desc3_3)
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc3_4 = {"subtask_id": "subtask_3.4", "instruction": cot_sc_instruction.format("phonons"), "context": ["user query", "response of subtask_1", "response of subtask_2"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking_i, answer_i = await cot_agents[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction.format("phonons"), is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, identify broken symmetry for phonons, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers.append(answer_i.content)
        thinkingmapping[answer_i.content] = thinking_i
        answermapping[answer_i.content] = answer_i
    answer34_content = Counter(possible_answers).most_common(1)[0][0]
    thinking3_4 = thinkingmapping[answer34_content]
    answer3_4 = answermapping[answer34_content]
    sub_tasks.append(f"Sub-task 3.4 output: thinking - {thinking3_4.content}; answer - {answer3_4.content}")
    print("Step 3.4: ", sub_tasks[-1])
    subtask_desc3_4['response'] = {"thinking": thinking3_4, "answer": answer3_4}
    logs.append(subtask_desc3_4)
    cot_reflect_instruction = "Sub-task 4: Compare the identified symmetries for magnons, skyrmions, pions, and phonons and determine which one has no associated spontaneously broken continuous symmetry."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    inputs4 = [taskInfo, thinking3_1, answer3_1, thinking3_2, answer3_2, thinking3_3, answer3_3, thinking3_4, answer3_4]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_reflect_instruction, "context": ["user query", "results of subtasks 3.1-3.4"], "agent_collaboration": "Reflexion"}
    thinking4, answer4 = await cot_agent4(inputs4, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, comparing symmetries, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4], "Please review the comparison and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content == "True":
            break
        inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(inputs4, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refining comparison, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    cot_instruction5 = "Sub-task 5: Map the particle identified in Sub-task 4 to the corresponding answer choice letter (A, B, C, or D)."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_instruction5, "context": ["user query", "answer of subtask_4"], "agent_collaboration": "CoT"}
    thinking5, answer5 = await cot_agent5([taskInfo, answer4], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, mapping to choice letter, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs