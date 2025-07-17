async def forward_176(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_inst1 = "Sub-task 1: Extract the given parameters R1/R2, M1/M2, identical peak wavelength implying T1/T2, and radial velocities v1 and v2 from the query."
    N = self.max_sc
    cot_sc_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings1 = []
    possible_answers1 = []
    sub_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_inst1,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking1_i, answer1_i = await cot_sc_agents1[i]([taskInfo], cot_sc_inst1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents1[i].id}, extracting parameters, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_dec1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_inst1 = "Given all the above thinkings and answers, synthesize and choose the most consistent extracted parameters."
    thinking1, answer1 = await final_dec1([taskInfo] + possible_thinkings1 + possible_answers1, final_inst1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    sub_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(sub_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_inst2 = "Sub-task 2: Based on R1/R2 = 1.5 and T1/T2 = 1, compute the baseline luminosity ratio L1/L2 using L ∝ R^2 T^4."
    cot_sc_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings2 = []
    possible_answers2 = []
    sub_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_inst2,"context":["user query","response of subtask_1"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking2_i, answer2_i = await cot_sc_agents2[i]([taskInfo, thinking1, answer1], cot_sc_inst2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents2[i].id}, computing baseline ratio, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_dec2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_inst2 = "Given all the above thinkings and answers, synthesize and choose the most consistent baseline luminosity ratio L1/L2."
    thinking2, answer2 = await final_dec2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, final_inst2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    sub_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(sub_desc2)
    print("Step 2: ", sub_tasks[-1])
    reflect_inst3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_inst3 = "Sub-task 3: Compute the Doppler shift correction for Star_2, calculate v2/c, derive the corrected T2_emit relative to observed T2, and find T1/T2_emit." + reflect_inst3
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    sub_desc3 = {"subtask_id":"subtask_3","instruction":cot_reflect_inst3,"context":["user query","thinking of subtask_1","answer of subtask_1","thinking of subtask_2","answer of subtask_2"],"agent_collaboration":"Reflexion"}
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_inst3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, initial Doppler correction, thinking: {thinking3.content}; answer: {answer3.content}")
    N_max = self.max_round
    for i in range(N_max):
        critic_inst3 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], critic_inst3, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct flag: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_inst3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refined Doppler correction, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    sub_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(sub_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_sc_inst4 = "Sub-task 4: Using R1/R2 = 1.5 and the corrected T1/T2_emit from subtask 3, compute the adjusted luminosity ratio L1/L2 and select the nearest factor among ~2.25, ~2.35, ~2.32, ~2.23."
    cot_sc_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings4 = []
    possible_answers4 = []
    sub_desc4 = {"subtask_id":"subtask_4","instruction":cot_sc_inst4,"context":["user query","thinking of subtask_2","answer of subtask_2","thinking of subtask_3","answer of subtask_3"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking4_i, answer4_i = await cot_sc_agents4[i]([taskInfo, thinking2, answer2, thinking3, answer3], cot_sc_inst4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents4[i].id}, computing adjusted ratio, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings4.append(thinking4_i)
        possible_answers4.append(answer4_i)
    final_dec4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_inst4 = "Given all the above thinkings and answers, synthesize and choose the most consistent adjusted luminosity ratio and corresponding multiple-choice option."
    thinking4, answer4 = await final_dec4([taskInfo, thinking2, answer2, thinking3, answer3] + possible_thinkings4 + possible_answers4, final_inst4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    sub_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(sub_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs