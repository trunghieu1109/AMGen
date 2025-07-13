async def forward_13(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: derive interior-chain projection with SC-CoT
    cot_sc_instruction = "Sub-task 1: Derive the interior-chain projection S_interior = (n-1)*2r*cos(B/2) by projecting consecutive center-to-center segments onto the angle bisector. Explicitly note each cosine projection step."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id": "stage1_subtask1", "instruction": cot_sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N1):
        thinking_i, answer_i = await cot_agents1[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings1.append(thinking_i)
        possible_answers1.append(answer_i)
    final_decision1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1([taskInfo] + possible_thinkings1 + possible_answers1,
                                               "Sub-task 1: Synthesize the most consistent derivation of S_interior.",
                                               is_sub_task=True)
    sub_tasks.append(f"Stage1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: solve for B by Debate
    debate_instruction2 = "Sub-task 1: Apply S_total = 2r(n-1)*cos(B/2) + 2r*sin(B/2) to both chains (r1=34,n1=8; r2=1,n2=2024), set equal, and solve for B." + 
                          " Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents2 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                       for role in self.debate_role]
    R2 = self.max_round
    all_think2 = [[] for _ in range(R2)]
    all_ans2 = [[] for _ in range(R2)]
    subtask_desc2 = {"subtask_id": "stage2_subtask1", "instruction": debate_instruction2,
                     "context": ["user query", thinking1, answer1], "agent_collaboration": "Debate"}
    for r in range(R2):
        for i, agent in enumerate(debate_agents2):
            if r == 0:
                thinking2_i, answer2_i = await agent([taskInfo, thinking1, answer1], debate_instruction2, r, is_sub_task=True)
            else:
                inputs2 = [taskInfo, thinking1, answer1] + all_think2[r-1] + all_ans2[r-1]
                thinking2_i, answer2_i = await agent(inputs2, debate_instruction2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
            all_think2[r].append(thinking2_i)
            all_ans2[r].append(answer2_i)
    final2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final2([taskInfo, thinking1, answer1] + all_think2[-1] + all_ans2[-1],
                                      "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
                                      is_sub_task=True)
    sub_tasks.append(f"Stage2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: express AB and BC via SC-CoT
    cot_sc_instruction3 = "Sub-task 1: Express AB and BC in terms of chain parameters and B by computing distance from B to first tangency points on AB and BC, including offsets r*cot(B/2)."
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                   for _ in range(N3)]
    possible_think3 = []
    possible_ans3 = []
    subtask_desc3 = {"subtask_id": "stage3_subtask1", "instruction": cot_sc_instruction3,
                     "context": ["user query", thinking2, answer2], "agent_collaboration": "SC_CoT"}
    for i in range(N3):
        thinking3_i, answer3_i = await cot_agents3[i]([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_think3.append(thinking3_i)
        possible_ans3.append(answer3_i)
    final3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final3([taskInfo, thinking2, answer2] + possible_think3 + possible_ans3,
                                      "Sub-task 1: Synthesize the expressions for AB and BC.",
                                      is_sub_task=True)
    sub_tasks.append(f"Stage3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 4: compute inradius via Reflexion
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction4 = "Sub-task 1: Compute the inradius using area=1/2*AB*BC*sin(B) and semiperimeter, then simplify to m/n and compute m+n." + reflect_inst
    critic_inst4 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {"subtask_id": "stage4_subtask1", "instruction": cot_reflect_instruction4,
                     "context": ["user query", thinking1, answer1, thinking2, answer2, thinking3, answer3],
                     "agent_collaboration": "Reflexion"}
    thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback4, correct4 = await critic_agent4(cot_inputs4 + [thinking4, answer4], critic_inst4, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content.strip() == "True":
            break
        cot_inputs4 += [thinking4, answer4, feedback4]
        thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction4, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refined thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Stage4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs