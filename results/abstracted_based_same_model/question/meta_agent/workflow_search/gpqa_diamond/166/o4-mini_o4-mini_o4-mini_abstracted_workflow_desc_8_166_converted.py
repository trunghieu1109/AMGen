async def forward_166(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Compute the normalization constant N for phi = -pi/4 and alpha = 0.5 using N = sqrt(1 + sin(2*phi)*exp(-2*alpha**2))."
    cot_agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, computing N, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction2 = "Sub-task 2: Truncate Fock-space at n_max=10 and compute Schrödinger cat state coefficients psi_n = [cos(phi)*<n|alpha> + sin(phi)*<n|-alpha>]/N for n=0..10, using phi=-pi/4, alpha=0.5 and N from Sub-task 1."
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible2 = []
    think2_map = {}
    ans2_map = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction2, "context": ["user query", "thinking of subtask_1", "answer of subtask_1"], "agent_collaboration": "SC_CoT"}
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, computing psi_n, thinking: {thinking2.content}; answer: {answer2.content}")
        possible2.append(answer2.content)
        think2_map[answer2.content] = thinking2
        ans2_map[answer2.content] = answer2
    choice2 = Counter(possible2).most_common(1)[0][0]
    thinking2 = think2_map[choice2]
    answer2 = ans2_map[choice2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction3 = "Sub-task 3: Assemble the truncated density matrix rho_mn = psi_m * conj(psi_n) for m,n=0..10 using coefficients from Sub-task 2."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_instruction3, "context": ["user query", "thinking of subtask_2", "answer of subtask_2"], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, assembling rho, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_reflect_instruction4 = "Sub-task 4: Verify that rho has trace equal to 1 (within tolerance) and is positive semidefinite. If checks fail, refine the assembly."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs4 = [taskInfo, thinking3, answer3]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_reflect_instruction4, "context": ["user query", "thinking of subtask_3", "answer of subtask_3"], "agent_collaboration": "Reflexion"}
    thinking4, answer4 = await cot_agent4(inputs4, cot_reflect_instruction4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, verifying rho, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4], "Review the verification of rho and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content == "True":
            break
        inputs4 += [thinking4, answer4, feedback4]
        thinking4, answer4 = await cot_agent4(inputs4, cot_reflect_instruction4, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refined verifying rho, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction5 = "Sub-task 5: Compute first moments <x>, <p> and covariance matrix elements V_xx, V_pp, V_xp from rho for m,n=0..10."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_instruction5, "context": ["user query", "thinking of subtask_3", "answer of subtask_3"], "agent_collaboration": "CoT"}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking3, answer3], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, computing moments, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_sc_instruction6 = "Sub-task 6: Construct reference Gaussian state tau by matching its covariance matrix from Sub-task 5 and setting its mean to (<x>, <p>)."
    cot_agents6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible6 = []
    think6_map = {}
    ans6_map = {}
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_sc_instruction6, "context": ["user query", "thinking of subtask_5", "answer of subtask_5"], "agent_collaboration": "SC_CoT"}
    for i in range(self.max_sc):
        thinking6, answer6 = await cot_agents6[i]([taskInfo, thinking5, answer5], cot_sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents6[i].id}, constructing tau, thinking: {thinking6.content}; answer: {answer6.content}")
        possible6.append(answer6.content)
        think6_map[answer6.content] = thinking6
        ans6_map[answer6.content] = answer6
    choice6 = Counter(possible6).most_common(1)[0][0]
    thinking6 = think6_map[choice6]
    answer6 = ans6_map[choice6]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    cot_sc_instruction7 = "Sub-task 7: Build the truncated Gaussian density matrix tau_mn in Fock basis up to n_max=10 using covariance and mean from Sub-task 6."
    cot_agents7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible7 = []
    think7_map = {}
    ans7_map = {}
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_sc_instruction7, "context": ["user query", "thinking of subtask_6", "answer of subtask_6"], "agent_collaboration": "SC_CoT"}
    for i in range(self.max_sc):
        thinking7, answer7 = await cot_agents7[i]([taskInfo, thinking6, answer6], cot_sc_instruction7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents7[i].id}, building tau matrix, thinking: {thinking7.content}; answer: {answer7.content}")
        possible7.append(answer7.content)
        think7_map[answer7.content] = thinking7
        ans7_map[answer7.content] = answer7
    choice7 = Counter(possible7).most_common(1)[0][0]
    thinking7 = think7_map[choice7]
    answer7 = ans7_map[choice7]
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    cot_reflect_instruction8 = "Sub-task 8: Verify that tau has trace equal to 1 and is positive semidefinite. If checks fail, refine the construction."
    cot_agent8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs8 = [taskInfo, thinking7, answer7]
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": cot_reflect_instruction8, "context": ["user query", "thinking of subtask_7", "answer of subtask_7"], "agent_collaboration": "Reflexion"}
    thinking8, answer8 = await cot_agent8(inputs8, cot_reflect_instruction8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent8.id}, verifying tau, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(self.max_round):
        feedback8, correct8 = await critic_agent8([taskInfo, thinking8, answer8], "Review the verification of tau and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent8.id}, feedback: {feedback8.content}; correct: {correct8.content}")
        if correct8.content == "True":
            break
        inputs8 += [thinking8, answer8, feedback8]
        thinking8, answer8 = await cot_agent8(inputs8, cot_reflect_instruction8, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent8.id}, refined verifying tau, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    cot_sc_instruction9 = "Sub-task 9: Diagonalize rho to get eigenvalues and compute S_rho = -sum(lambda_i ln lambda_i)."
    cot_agents9 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible9 = []
    think9_map = {}
    ans9_map = {}
    subtask_desc9 = {"subtask_id": "subtask_9", "instruction": cot_sc_instruction9, "context": ["user query", "thinking of subtask_4", "answer of subtask_4"], "agent_collaboration": "SC_CoT"}
    for i in range(self.max_sc):
        thinking9, answer9 = await cot_agents9[i]([taskInfo, thinking4, answer4], cot_sc_instruction9, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents9[i].id}, computing S_rho, thinking: {thinking9.content}; answer: {answer9.content}")
        possible9.append(answer9.content)
        think9_map[answer9.content] = thinking9
        ans9_map[answer9.content] = answer9
    choice9 = Counter(possible9).most_common(1)[0][0]
    thinking9 = think9_map[choice9]
    answer9 = ans9_map[choice9]
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    cot_sc_instruction10 = "Sub-task 10: Diagonalize tau to get eigenvalues and compute S_tau = -sum(mu_j ln mu_j)."
    cot_agents10 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible10 = []
    think10_map = {}
    ans10_map = {}
    subtask_desc10 = {"subtask_id": "subtask_10", "instruction": cot_sc_instruction10, "context": ["user query", "thinking of subtask_8", "answer of subtask_8"], "agent_collaboration": "SC_CoT"}
    for i in range(self.max_sc):
        thinking10, answer10 = await cot_agents10[i]([taskInfo, thinking8, answer8], cot_sc_instruction10, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents10[i].id}, computing S_tau, thinking: {thinking10.content}; answer: {answer10.content}")
        possible10.append(answer10.content)
        think10_map[answer10.content] = thinking10
        ans10_map[answer10.content] = answer10
    choice10 = Counter(possible10).most_common(1)[0][0]
    thinking10 = think10_map[choice10]
    answer10 = ans10_map[choice10]
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])
    cot_instruction11 = "Sub-task 11: Calculate non-Gaussianity Δ_b = trace(rho ln rho) - trace(tau ln tau) using results from Sub-task 9 and Sub-task 10."
    cot_agent11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc11 = {"subtask_id": "subtask_11", "instruction": cot_instruction11, "context": ["user query", "thinking of subtask_9", "answer of subtask_9", "thinking of subtask_10", "answer of subtask_10"], "agent_collaboration": "CoT"}
    thinking11, answer11 = await cot_agent11([taskInfo, thinking9, answer9, thinking10, answer10], cot_instruction11, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent11.id}, calculating Δ_b, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    subtask_desc11['response'] = {"thinking": thinking11, "answer": answer11}
    logs.append(subtask_desc11)
    print("Step 11: ", sub_tasks[-1])
    debate_instruction12 = "Sub-task 12: Numerically evaluate Δ_b for phi = -pi/4 and alpha = 0.5, then compare against choices [2.48, 0, 1.38, 0.25] and select the correct letter A–D."
    debate_agents12 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking12 = [[] for _ in range(self.max_round)]
    all_answer12 = [[] for _ in range(self.max_round)]
    subtask_desc12 = {"subtask_id": "subtask_12", "instruction": debate_instruction12, "context": ["user query", "thinking of subtask_11", "answer of subtask_11"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents12):
            if r == 0:
                thinking12, answer12 = await agent([taskInfo, thinking11, answer11], debate_instruction12, r, is_sub_task=True)
            else:
                inputs12 = [taskInfo, thinking11, answer11] + all_thinking12[r-1] + all_answer12[r-1]
                thinking12, answer12 = await agent(inputs12, debate_instruction12, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking12.content}; answer: {answer12.content}")
            all_thinking12[r].append(thinking12)
            all_answer12[r].append(answer12)
    final_decision_agent12 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking12, answer12 = await final_decision_agent12([taskInfo] + all_thinking12[-1] + all_answer12[-1], "Sub-task 12: Make final selection among choices A–D for Δ_b.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent12.id}, thinking: {thinking12.content}; answer: {answer12.content}")
    sub_tasks.append(f"Sub-task 12 output: thinking - {thinking12.content}; answer - {answer12.content}")
    subtask_desc12['response'] = {"thinking": thinking12, "answer": answer12}
    logs.append(subtask_desc12)
    print("Step 12: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking12, answer12, sub_tasks, agents)
    return final_answer, logs