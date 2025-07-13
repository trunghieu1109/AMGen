async def forward_176(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: SC_CoT
    sc1_instr = "Sub-task 1: Extract and summarize all given parameters including R1/R2=1.5, M1/M2=1.5, observed λ_max, radial velocities, and note ambiguity on Doppler correction."
    N_sc = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    thinkers1, answers1 = [], []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents1[i]([taskInfo], sc1_instr, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, extracting parameters, thinking: {thinking_i.content}; answer: {answer_i.content}")
        thinkers1.append(thinking_i)
        answers1.append(answer_i)
    final_decision1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final1_instr = "Given all above thoughts and answers, choose the most consistent summary of parameters."
    thinking1, answer1 = await final_decision1([taskInfo] + thinkers1 + answers1, final1_instr, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append({"subtask_id":"subtask_1","instruction":sc1_instr,"context":["user query"],"agent_collaboration":"SC_CoT","response":{"thinking":thinking1,"answer":answer1})
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Debate
    debate2_instr = "Sub-task 2: Clarify Doppler correction by defining scenario A (λ_max intrinsic) and B (λ_max observed), and decide which is implied." + 
                    " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    rounds2 = self.max_round
    all_think2 = [[] for _ in range(rounds2)]
    all_ans2 = [[] for _ in range(rounds2)]
    for r in range(rounds2):
        for i, agent in enumerate(debate_agents2):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking1, answer1], debate2_instr, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1] + all_think2[r-1] + all_ans2[r-1]
                thinking_i, answer_i = await agent(inputs, debate2_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_think2[r].append(thinking_i)
            all_ans2[r].append(answer_i)
    final_deb2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final2_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking2, answer2 = await final_deb2([taskInfo, thinking1, answer1] + all_think2[-1] + all_ans2[-1], final2_instr, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append({"subtask_id":"subtask_2","instruction":debate2_instr,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"Debate","response":{"thinking":thinking2,"answer":answer2})
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: SC_CoT
    sc3_instr = "Sub-task 3: Identify and detail governing physics: L ∝ R^2 T^4, Wien’s law λ_max ∝ 1/T, and relativistic Doppler shift formula, integrating Doppler effect for scenario B."
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    thinkers3, answers3 = [], []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents3[i]([taskInfo, thinking1, answer1, thinking2, answer2], sc3_instr, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, detailing physics, thinking: {thinking_i.content}; answer: {answer_i.content}")
        thinkers3.append(thinking_i)
        answers3.append(answer_i)
    final_decision3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final3_instr = "Given all above thoughts and answers, synthesize the governing physics details into a coherent description."
    thinking3, answer3 = await final_decision3([taskInfo, thinking1, answer1, thinking2, answer2] + thinkers3 + answers3, final3_instr, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append({"subtask_id":"subtask_3","instruction":sc3_instr,"context":["user query","thinking of subtask 1","answer of subtask 1","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"SC_CoT","response":{"thinking":thinking3,"answer":answer3})
    print("Step 3: ", sub_tasks[-1])

    # Stage 4a: CoT
    cot4a_instr = "Sub-task 4a: Derive expression for L1/L2 under scenario A using R1/R2=1.5 and T1=T2."
    cot4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4a, answer4a = await cot4a([taskInfo, thinking3, answer3], cot4a_instr, is_sub_task=True)
    agents.append(f"CoT agent {cot4a.id}, deriving naive ratio, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    logs.append({"subtask_id":"subtask_4a","instruction":cot4a_instr,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"CoT","response":{"thinking":thinking4a,"answer":answer4a})
    print("Step 4a: ", sub_tasks[-1])

    # Stage 4b: CoT
    cot4b_instr = "Sub-task 4b: Derive expression for L1/L2 under scenario B using Doppler correction T2_rest = T_obs×(1+v/c) and L ∝ R^2 T^4."
    cot4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4b, answer4b = await cot4b([taskInfo, thinking3, answer3], cot4b_instr, is_sub_task=True)
    agents.append(f"CoT agent {cot4b.id}, deriving Doppler-corrected ratio, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    logs.append({"subtask_id":"subtask_4b","instruction":cot4b_instr,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"CoT","response":{"thinking":thinking4b,"answer":answer4b})
    print("Step 4b: ", sub_tasks[-1])

    # Stage 5: CoT
    cot5_instr = "Sub-task 5: Compute numerical values: naive ratio = 2.25 and Doppler-corrected ratio using v=700 km/s and c=3×10^5 km/s."
    cot5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot5([taskInfo, thinking4a, answer4a, thinking4b, answer4b], cot5_instr, is_sub_task=True)
    agents.append(f"CoT agent {cot5.id}, computing ratios, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    logs.append({"subtask_id":"subtask_5","instruction":cot5_instr,"context":["user query","thinking of subtask 4a","answer of subtask 4a","thinking of subtask 4b","answer of subtask 4b"],"agent_collaboration":"CoT","response":{"thinking":thinking5,"answer":answer5})
    print("Step 5: ", sub_tasks[-1])

    # Stage 6: Debate
    debate6_instr = "Sub-task 6: Compare the two ratios and select the multiple-choice option matching the physically correct case based on scenario justification." + 
                    " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    rounds6 = self.max_round
    all_think6 = [[] for _ in range(rounds6)]
    all_ans6 = [[] for _ in range(rounds6)]
    for r in range(rounds6):
        for i, agent in enumerate(debate_agents6):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking2, answer2, thinking5, answer5], debate6_instr, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2, thinking5, answer5] + all_think6[r-1] + all_ans6[r-1]
                thinking_i, answer_i = await agent(inputs, debate6_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_think6[r].append(thinking_i)
            all_ans6[r].append(answer_i)
    final_deb6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final6_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking6, answer6 = await final_deb6([taskInfo, thinking2, answer2, thinking5, answer5] + all_think6[-1] + all_ans6[-1], final6_instr, is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    logs.append({"subtask_id":"subtask_6","instruction":debate6_instr,"context":["user query","thinking of subtask 2","answer of subtask 2","thinking of subtask 5","answer of subtask 5"],"agent_collaboration":"Debate","response":{"thinking":thinking6,"answer":answer6})
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs