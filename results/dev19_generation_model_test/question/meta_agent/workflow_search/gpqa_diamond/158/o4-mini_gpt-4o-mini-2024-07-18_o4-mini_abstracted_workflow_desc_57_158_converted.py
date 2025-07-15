async def forward_158(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1, Sub-task 1: Extract observational data (SC-CoT)
    instr1 = "Sub-task 1: Extract and summarize the observational data: quasar spectral peak at 790 nm and the significant flux drop at wavelengths below 790 nm."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_th1, possible_ans1 = [], []
    desc1 = {"subtask_id": "subtask_1", "instruction": instr1, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N1):
        t1, a1 = await cot_agents1[i]([taskInfo], instr1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, extracting observational data, thinking: {t1.content}; answer: {a1.content}")
        possible_th1.append(t1)
        possible_ans1.append(a1)
    final_dec1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    t1_final, a1_final = await final_dec1([taskInfo] + possible_th1 + possible_ans1,
        "Sub-task 1: Synthesize and choose the most consistent summary of observational data.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {t1_final.content}; answer - {a1_final.content}")
    desc1['response'] = {"thinking": t1_final, "answer": a1_final}
    logs.append(desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1, Sub-task 2: Extract cosmological parameters (SC-CoT)
    instr2 = "Sub-task 2: Extract and summarize the cosmological model parameters: H0 = 70 km/s/Mpc, Ω_m = 0.3, Ω_Λ = 0.7, flat universe."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_th2, possible_ans2 = [], []
    desc2 = {"subtask_id": "subtask_2", "instruction": instr2, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        t2, a2 = await cot_agents2[i]([taskInfo], instr2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, extracting cosmological parameters, thinking: {t2.content}; answer: {a2.content}")
        possible_th2.append(t2)
        possible_ans2.append(a2)
    final_dec2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    t2_final, a2_final = await final_dec2([taskInfo] + possible_th2 + possible_ans2,
        "Sub-task 2: Synthesize and choose the most consistent summary of cosmological parameters.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {t2_final.content}; answer - {a2_final.content}")
    desc2['response'] = {"thinking": t2_final, "answer": a2_final}
    logs.append(desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2, Sub-task 3: Validate Lyman-α break and compute z (Debate)
    debate_instr3 = (
        "Sub-task 3: Validate that the flux drop corresponds to the Lyman-α break at 121.6 nm rest-frame and compute the quasar redshift z = λ_obs/λ_rest - 1. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                      for role in self.debate_role]
    all_th3 = [[] for _ in range(self.max_round)]
    all_ans3 = [[] for _ in range(self.max_round)]
    desc3 = {"subtask_id": "subtask_3", "instruction": debate_instr3,
             "context": ["user query", "subtask_1", "subtask_2"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents3):
            if r == 0:
                t3, a3 = await agent([taskInfo, t1_final, a1_final, t2_final, a2_final], debate_instr3, r, is_sub_task=True)
            else:
                t3, a3 = await agent([taskInfo, t1_final, a1_final, t2_final, a2_final] + all_th3[r-1] + all_ans3[r-1], debate_instr3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t3.content}; answer: {a3.content}")
            all_th3[r].append(t3)
            all_ans3[r].append(a3)
    final_dec3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    t3_final, a3_final = await final_dec3(
        [taskInfo, t1_final, a1_final, t2_final, a2_final] + all_th3[-1] + all_ans3[-1],
        "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {t3_final.content}; answer - {a3_final.content}")
    desc3['response'] = {"thinking": t3_final, "answer": a3_final}
    logs.append(desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 3, Sub-task 4: Compute comoving distance (Reflexion)
    reflect_inst4 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    instr4 = (
        "Sub-task 4: Compute the comoving distance D_c(z) = c ∫₀^z dz'/H(z') for z from Sub-task 3 using H0=70 km/s/Mpc, Ω_m=0.3, Ω_Λ=0.7, flat universe. "
        + reflect_inst4
    )
    cot4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs4 = [taskInfo, t1_final, a1_final, t2_final, a2_final, t3_final, a3_final]
    desc4 = {"subtask_id": "subtask_4", "instruction": instr4,
             "context": ["user query", "subtask_1", "subtask_2", "subtask_3"], "agent_collaboration": "Reflexion"}
    thinking4, answer4 = await cot4(inputs4, instr4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback4, correct4 = await critic4(
            [taskInfo, thinking4, answer4],
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", 
            i, is_sub_task=True
        )
        agents.append(f"Critic agent {critic4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content == "True":
            break
        inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot4(inputs4, instr4, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot4.id}, refinement round {i+1}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 4, Sub-task 5: Choose nearest comoving distance (SC-CoT)
    instr5 = "Sub-task 5: Compare the calculated comoving distance with answer choices (6, 7, 8, 9 Gpc) and select the nearest value."
    N5 = self.max_sc
    cot_agents5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_th5, possible_ans5 = [], []
    desc5 = {"subtask_id": "subtask_5", "instruction": instr5,
             "context": ["user query", "subtask_4"], "agent_collaboration": "SC_CoT"}
    for i in range(N5):
        t5, a5 = await cot_agents5[i]([taskInfo, thinking4, answer4], instr5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents5[i].id}, comparing distances, thinking: {t5.content}; answer: {a5.content}")
        possible_th5.append(t5)
        possible_ans5.append(a5)
    final_dec5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    t5_final, a5_final = await final_dec5([taskInfo, thinking4, answer4] + possible_th5 + possible_ans5,
        "Sub-task 5: Synthesize and choose the most consistent selection among (6,7,8,9 Gpc).", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {t5_final.content}; answer - {a5_final.content}")
    desc5['response'] = {"thinking": t5_final, "answer": a5_final}
    logs.append(desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(t5_final, a5_final, sub_tasks, agents)
    return final_answer, logs