async def forward_166(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Sub-task 1: Compute normalization constant N = sqrt(1 + sin(2*phi)*exp(-2*alpha^2)) for phi = -pi/4 and alpha = 0.5, return numeric value to three decimal places."
    cot_agent1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_instruction1,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, computing N, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    N2 = self.max_sc
    cot_sc_instruction2 = "Sub-task 2: Using N from Sub-task 1, construct normalized cat state |psi> = (cos(phi)|alpha> + sin(phi)|-alpha>)/N and represent it in coherent-state basis."
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible2 = []
    thinkingmap2 = {}
    answermap2 = {}
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction2,"context":["user query","thinking of subtask_1","answer of subtask_1"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents2:
        t2, a2 = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, constructing |psi>, thinking: {t2.content}; answer: {a2.content}")
        possible2.append(a2.content)
        thinkingmap2[a2.content] = t2
        answermap2[a2.content] = a2
    answer2_content = Counter(possible2).most_common(1)[0][0]
    thinking2 = thinkingmap2[answer2_content]
    answer2 = answermap2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction3 = "Sub-task 3: Form density matrix rho = |psi><psi| in basis of |alpha>,|-alpha> or truncated Fock basis."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_instruction3,"context":["user query","thinking of subtask_2","answer of subtask_2"],"agent_collaboration":"CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, forming rho, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    N4 = self.max_sc
    cot_sc_instruction4 = "Sub-task 4: Compute first-order moments <x>,<p> and covariance matrix V of rho; provide formulas, intermediate values, and final numeric entries to three decimals."
    cot_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible4 = []
    thinkingmap4 = {}
    answermap4 = {}
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_sc_instruction4,"context":["user query","thinking of subtask_3","answer of subtask_3"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents4:
        t4, a4 = await agent([taskInfo, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, computing moments, thinking: {t4.content}; answer: {a4.content}")
        possible4.append(a4.content)
        thinkingmap4[a4.content] = t4
        answermap4[a4.content] = a4
    answer4_content = Counter(possible4).most_common(1)[0][0]
    thinking4 = thinkingmap4[answer4_content]
    answer4 = answermap4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_reflect_instruction5 = "Sub-task 5: Construct reference Gaussian state tau with same mean (<x>,<p>) and covariance V; express density operator parameters explicitly."
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":cot_reflect_instruction5,"context":["user query","thinking of subtask_4","answer of subtask_4"],"agent_collaboration":"Reflexion"}
    cot_inputs5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, constructing tau, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(self.max_round):
        feedback5, correct5 = await critic_agent5([taskInfo, thinking5, answer5], "Please review the constructed tau and provide corrections.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback: {feedback5.content}; correct: {correct5.content}")
        if correct5.content == "True":
            break
        cot_inputs5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction5, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refining tau, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction6 = "Sub-task 6: Evaluate trace(rho ln rho) for pure state rho and show trace(rho ln rho)=0 with brief proof."
    cot_agent6 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {"subtask_id":"subtask_6","instruction":cot_instruction6,"context":["user query","thinking of subtask_3","answer of subtask_3"],"agent_collaboration":"CoT"}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking3, answer3], cot_instruction6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, evaluating trace(rho ln rho), thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    debate_instruction7 = "Sub-task 7: Compute symplectic eigenvalues of V and Gaussian entropy S(tau), then return trace(tau ln tau) = -S(tau) to three decimals."
    debate_agents7 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking7 = [[] for _ in range(self.max_round)]
    all_answer7 = [[] for _ in range(self.max_round)]
    subtask_desc7 = {"subtask_id":"subtask_7","instruction":debate_instruction7,"context":["user query","thinking of subtask_5","answer of subtask_5"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents7:
            if r == 0:
                t7, a7 = await agent([taskInfo, thinking5, answer5], debate_instruction7, r, is_sub_task=True)
            else:
                t7, a7 = await agent([taskInfo, thinking5, answer5] + all_thinking7[r-1] + all_answer7[r-1], debate_instruction7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing trace(tau ln tau), thinking: {t7.content}; answer: {a7.content}")
            all_thinking7[r].append(t7)
            all_answer7[r].append(a7)
    final_decision_agent7 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Decide on numeric value of trace(tau ln tau).", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent7.id}, final decision on trace(tau ln tau), thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction8 = "Sub-task 8: Compute non-Gaussianity Delta_b = trace(rho ln rho) - trace(tau ln tau) using outputs of Subtasks 6 and 7 to three decimals."
    cot_agent8 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {"subtask_id":"subtask_8","instruction":cot_instruction8,"context":["user query","answer of subtask_6","answer of subtask_7"],"agent_collaboration":"CoT"}
    thinking8, answer8 = await cot_agent8([taskInfo, answer6, answer7], cot_instruction8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, computing Delta_b, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8["response"] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    debate_instruction9 = "Sub-task 9: Compare Delta_b to {A:2.48,B:0,C:1.38,D:0.25} and select closest match letter."
    debate_agents9 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking9 = [[] for _ in range(self.max_round)]
    all_answer9 = [[] for _ in range(self.max_round)]
    subtask_desc9 = {"subtask_id":"subtask_9","instruction":debate_instruction9,"context":["user query","thinking of subtask_8","answer of subtask_8"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents9:
            if r == 0:
                t9, a9 = await agent([taskInfo, thinking8, answer8], debate_instruction9, r, is_sub_task=True)
            else:
                t9, a9 = await agent([taskInfo, thinking8, answer8] + all_thinking9[r-1] + all_answer9[r-1], debate_instruction9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting choice, thinking: {t9.content}; answer: {a9.content}")
            all_thinking9[r].append(t9)
            all_answer9[r].append(a9)
    final_decision_agent9 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Make final decision on non-Gaussianity choice.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent9.id}, final decision on choice, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9["response"] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs