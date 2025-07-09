async def forward_173(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Determine the rest masses m_small and m_large of the two fission fragments in GeV/c², given M=300 GeV and m_large=2*m_small and m_small+m_large=0.99*M."
    cot_agent1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, computing rest masses, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: Compute the total kinetic energy available E_available = 0.01*M in GeV."
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query","thinking of subtask 1","answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, computing E_available, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction3a = "Sub-task 3a: Calculate the classical kinetic energy T1_classical = E_available * (m_small/(m_small+m_large)) in GeV."
    cot_agent3a = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3a = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs3a = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3a = {"subtask_id": "subtask_3a", "instruction": cot_instruction3a, "context": ["user query","thinking of subtask 1","answer of subtask 1","thinking of subtask 2","answer of subtask 2"], "agent_collaboration": "Reflexion"}
    thinking3a, answer3a = await cot_agent3a(cot_inputs3a, cot_instruction3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3a.id}, computing T1_classical, thinking: {thinking3a.content}; answer: {answer3a.content}")
    for i in range(N_max):
        feedback3a, correct3a = await critic_agent3a([taskInfo, thinking3a, answer3a], "Review the non-relativistic kinetic energy calculation T1_classical and confirm correctness or provide corrections.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3a.id}, feedback: {feedback3a.content}; correction flag: {correct3a.content}")
        if correct3a.content == "True":
            break
        cot_inputs3a.extend([thinking3a, answer3a, feedback3a])
        thinking3a, answer3a = await cot_agent3a(cot_inputs3a, cot_instruction3a, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3a.id}, refining T1_classical, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3: ", sub_tasks[-1])
    cot_sc_instruction3b = "Sub-task 3b: Derive the common momentum p by solving sqrt(p^2+m_small^2)+sqrt(p^2+m_large^2)=M, then compute T1_relativistic = sqrt(p^2+m_large^2)-m_large in GeV."
    cot_agents3b = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers3b = []
    thinkingmapping3b = {}
    answermapping3b = {}
    subtask_desc3b = {"subtask_id": "subtask_3b", "instruction": cot_sc_instruction3b, "context": ["user query","thinking of subtask 1","answer of subtask 1","thinking of subtask 2","answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking3b, answer3b = await cot_agents3b[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3b[i].id}, computing T1_relativistic, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_answers3b.append(answer3b.content)
        thinkingmapping3b[answer3b.content] = thinking3b
        answermapping3b[answer3b.content] = answer3b
    answer3b_content = Counter(possible_answers3b).most_common(1)[0][0]
    thinking3b = thinkingmapping3b[answer3b_content]
    answer3b = answermapping3b[answer3b_content]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction4 = "Sub-task 4: Compute ΔT = T1_relativistic - T1_classical and convert ΔT from GeV to MeV."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query","thinking of subtask 3a","answer of subtask 3a","thinking of subtask 3b","answer of subtask 3b"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3a, answer3a, thinking3b, answer3b], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, computing ΔT, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 5: ", sub_tasks[-1])
    debate_instruction5 = "Sub-task 5: Compare the calculated ΔT in MeV to the provided choices (10, 5, 2, 20 MeV) and select the matching answer letter (A, B, C, or D)."
    debate_agents5 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max5)]
    all_answer5 = [[] for _ in range(N_max5)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction5, "context": ["user query","thinking of subtask 4","answer of subtask 4"], "agent_collaboration": "Debate"}
    for r in range(N_max5):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on ΔT comparison.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs