async def forward_154(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Encode the quantum state psi and operator P_z in suitable form for calculation."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, encoding psi and P_z, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: Compute the expectation value <P_z> = psi^† P_z psi using the encoded psi and P_z."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers2 = []
    thinking_map2 = {}
    answer_map2 = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query","response of subtask_1"], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, computing <P_z>, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers2.append(answer2.content)
        thinking_map2[answer2.content] = thinking2
        answer_map2[answer2.content] = answer2
    answer2_content = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinking_map2[answer2_content]
    answer2 = answer_map2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_instruction3 = "Sub-task 3: Compute the expectation value <P_z^2> = psi^† P_z^2 psi using the encoded psi and P_z."
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_answers3 = []
    thinking_map3 = {}
    answer_map3 = {}
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction3, "context": ["user query","response of subtask_1"], "agent_collaboration": "SC_CoT"}
    for i in range(N3):
        thinking3, answer3 = await cot_agents3[i]([taskInfo, thinking1, answer1], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, computing <P_z^2>, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers3.append(answer3.content)
        thinking_map3[answer3.content] = thinking3
        answer_map3[answer3.content] = answer3
    answer3_content = Counter(possible_answers3).most_common(1)[0][0]
    thinking3 = thinking_map3[answer3_content]
    answer3 = answer_map3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot4_instruction = "Sub-task 4: Calculate variance var(P_z) = <P_z^2> - (<P_z>)^2 from the results of Sub-task 2 and Sub-task 3."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot4_instruction, "context": ["user query","response of subtask_2","response of subtask_3"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking2, answer2, thinking3, answer3], cot4_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, calculating variance, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot5_instruction = "Sub-task 5: Compute the uncertainty ΔP_z = sqrt(var(P_z)) from the variance result."
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot5_instruction, "context": ["user query","response of subtask_4"], "agent_collaboration": "CoT"}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], cot5_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, computing uncertainty, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    debate_instruction6 = "Sub-task 6: Match the numerical result for ΔP_z against the provided choices and return the corresponding letter option."
    debate_agents6 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N6 = self.max_round
    all_thinking6 = [[] for _ in range(N6)]
    all_answer6 = [[] for _ in range(N6)]
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": debate_instruction6, "context": ["user query","response of subtask_5"], "agent_collaboration": "Debate"}
    for r in range(N6):
        for i, agent in enumerate(debate_agents6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction6, r, is_sub_task=True)
            else:
                input_infos6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos6, debate_instruction6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, matching choice, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent6 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the choice letter.", is_sub_task=True)
    agents.append(f"Final Decision agent, choosing letter, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs