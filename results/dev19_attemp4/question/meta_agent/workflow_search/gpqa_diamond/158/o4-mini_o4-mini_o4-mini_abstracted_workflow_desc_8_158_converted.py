async def forward_158(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    debate_instr = "Sub-task 1: Determine which rest-frame Lyman transition (Lyα at 121.6 nm or the Lyman limit at 91.2 nm) causes the sharp break at 790 nm. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_round = self.max_round
    all_thinking1 = [[] for _ in range(N_round)]
    all_answer1 = [[] for _ in range(N_round)]
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":debate_instr,"context":["user query"],"agent_collaboration":"Debate"}
    for r in range(N_round):
        for agent in debate_agents:
            if r == 0:
                thinking1, answer1 = await agent([taskInfo], debate_instr, r, is_sub_task=True)
            else:
                thinking1, answer1 = await agent([taskInfo] + all_thinking1[r-1] + all_answer1[r-1], debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking1[r].append(thinking1)
            all_answer1[r].append(answer1)
    final_decision1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = "Sub-task 1: Determine which rest-frame Lyman transition causes the break. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking1, answer1 = await final_decision1([taskInfo] + all_thinking1[-1] + all_answer1[-1], final_instr1, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision1.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction2 = "Sub-task 2: Calculate the quasar's redshift z using z = (λ_obs/λ_rest) - 1 based on the chosen rest-frame wavelength from Sub-task 1."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction2,"context":["user query","thinking of subtask_1","answer of subtask_1"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents2:
        thinking2_i, answer2_i = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_decision2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Sub-task 2: Synthesize and choose the most consistent and correct redshift solution. Given all the above thinking and answers, provide the best answer."
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, final_instr2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_instruction3 = "Sub-task 3: Compute the comoving distance D_C = (c/H0) ∫_0^z dz'/√[Ω_m(1+z')^3 + Ω_Λ] using H0=70 km/s/Mpc, Ω_m=0.3, Ω_Λ=0.7, based on the redshift z from Sub-task 2."
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_sc_instruction3,"context":["user query","thinking of subtask_2","answer of subtask_2"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents3:
        thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings3.append(thinking3_i)
        possible_answers3.append(answer3_i)
    final_decision3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Sub-task 3: Synthesize and choose the most consistent and correct comoving distance calculation. Given all the above thinking and answers, provide the best answer."
    thinking3, answer3 = await final_decision3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3, final_instr3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction4 = "Sub-task 4: Compare the computed comoving distance to choices 6 Gpc, 7 Gpc, 8 Gpc, 9 Gpc and select the closest value." + reflect_inst
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N4 = self.max_round
    inputs4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_reflect_instruction4,"context":["user query","thinking of subtask_3","answer of subtask_3"],"agent_collaboration":"Reflexion"}
    thinking4, answer4 = await cot_agent4(inputs4, cot_reflect_instruction4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N4):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content == "True":
            break
        inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(inputs4, cot_reflect_instruction4, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs