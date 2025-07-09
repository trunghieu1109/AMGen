async def forward_187(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Sub-task 1: Extract from the query the lattice constant (a = 10 Å), the rhombohedral angles (α = β = γ = 30°), and the Miller indices (h = k = l = 1)."
    cot_agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction1, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, extracting parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction2 = "Sub-task 2: Derive the metric tensor components for a rhombohedral lattice in terms of a and α, showing the step-by-step expressions for g_ij."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers2 = []
    thinking_mapping2 = {}
    answer_mapping2 = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction2, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, deriving metric tensor, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers2.append(answer2_i.content)
        thinking_mapping2[answer2_i.content] = thinking2_i
        answer_mapping2[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers2).most_common(1)[0][0]
    thinking2, answer2 = thinking_mapping2[answer2_content], answer_mapping2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_instruction3 = "Sub-task 3: From the metric tensor components, derive the general formula for the interplanar spacing d_hkl in a rhombohedral lattice, expressing it in terms of a, α, and (h, k, l)."
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_answers3 = []
    thinking_mapping3 = {}
    answer_mapping3 = {}
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction3, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for i in range(N3):
        thinking3_i, answer3_i = await cot_agents3[i]([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, deriving formula d_hkl, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_answers3.append(answer3_i.content)
        thinking_mapping3[answer3_i.content] = thinking3_i
        answer_mapping3[answer3_i.content] = answer3_i
    answer3_content = Counter(possible_answers3).most_common(1)[0][0]
    thinking3, answer3 = thinking_mapping3[answer3_content], answer_mapping3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction4 = "Sub-task 4: Compute numerical values of cos(α) and sin(α) for α = 30°, performing a mini-verification that cos²(α) + sin²(α) = 1."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking1, answer1], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, computing cos and sin, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction5 = "Sub-task 5: Substitute a = 10 Å and the numerical values of cos(α) and sin(α) into the metric tensor to compute its numerical components, showing intermediate checks for each entry."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_instruction5, "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "CoT"}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking2, answer2, thinking4, answer4], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, computing numerical tensor components, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_sc_instruction6 = "Sub-task 6: Plug the numerical metric tensor components and (h, k, l) = (1, 1, 1) into the general d_hkl formula, carrying out detailed arithmetic with intermediate validations for sums and square roots."
    N6 = self.max_sc
    cot_agents6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N6)]
    possible_answers6 = []
    thinking_mapping6 = {}
    answer_mapping6 = {}
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_sc_instruction6, "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 5", "answer of subtask 5"], "agent_collaboration": "SC_CoT"}
    for i in range(N6):
        thinking6_i, answer6_i = await cot_agents6[i]([taskInfo, thinking3, answer3, thinking5, answer5], cot_sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents6[i].id}, computing d_111 numerically, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
        possible_answers6.append(answer6_i.content)
        thinking_mapping6[answer6_i.content] = thinking6_i
        answer_mapping6[answer6_i.content] = answer6_i
    answer6_content = Counter(possible_answers6).most_common(1)[0][0]
    thinking6, answer6 = thinking_mapping6[answer6_content], answer_mapping6[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    cot_sc_instruction7 = "Sub-task 7: Independently re-derive the interplanar distance d_111 via an alternate route (e.g., special-case comparison) to confirm the result."
    N7 = self.max_sc
    cot_agents7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N7)]
    possible_answers7 = []
    thinking_mapping7 = {}
    answer_mapping7 = {}
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_sc_instruction7, "context": ["user query", "thinking of subtask 6", "answer of subtask 6"], "agent_collaboration": "SC_CoT"}
    for i in range(N7):
        thinking7_i, answer7_i = await cot_agents7[i]([taskInfo, thinking6, answer6], cot_sc_instruction7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents7[i].id}, self-consistency re-derivation, thinking: {thinking7_i.content}; answer: {answer7_i.content}")
        possible_answers7.append(answer7_i.content)
        thinking_mapping7[answer7_i.content] = thinking7_i
        answer_mapping7[answer7_i.content] = answer7_i
    answer7_content = Counter(possible_answers7).most_common(1)[0][0]
    thinking7, answer7 = thinking_mapping7[answer7_content], answer_mapping7[answer7_content]
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    debate_instruction8 = "Sub-task 8: Compare the verified interplanar distance against the provided options (9.54 Å, 8.95 Å, 9.08 Å, 10.05 Å) and select the closest match."
    debate_agents8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N8 = self.max_round
    all_thinking8 = [[] for _ in range(N8)]
    all_answer8 = [[] for _ in range(N8)]
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": debate_instruction8, "context": ["user query", "thinking of subtask 6", "answer of subtask 6", "thinking of subtask 7", "answer of subtask 7"], "agent_collaboration": "Debate"}
    for r in range(N8):
        for i, agent in enumerate(debate_agents8):
            if r == 0:
                thinking8_i, answer8_i = await agent([taskInfo, thinking6, answer6, thinking7, answer7], debate_instruction8, r, is_sub_task=True)
            else:
                inputs8 = [taskInfo, thinking6, answer6, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1]
                thinking8_i, answer8_i = await agent(inputs8, debate_instruction8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting choice, thinking: {thinking8_i.content}; answer: {answer8_i.content}")
            all_thinking8[r].append(thinking8_i)
            all_answer8[r].append(answer8_i)
    final_decision_agent8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent8([taskInfo] + all_thinking8[-1] + all_answer8[-1], "Sub-task 8: Make final decision on the closest interplanar distance.", is_sub_task=True)
    agents.append(f"Final Decision agent, final selection, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs