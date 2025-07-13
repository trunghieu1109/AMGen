async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: SC_CoT
    sc_instruction1 = "Sub-task 1: Parametrize z on the circle |z|=4 by setting z=4*exp(i*theta) and rewrite both (75+117i)*z and (96+144i)/z in exponential form."
    N1 = self.max_sc
    sc_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    poss_think1, poss_ans1 = [], []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":sc_instruction1,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking1_i, answer1_i = await sc_agents1[i]([taskInfo], sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents1[i].id}, parametrizing z, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        poss_think1.append(thinking1_i)
        poss_ans1.append(answer1_i)
    final_decision1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1([taskInfo] + poss_think1 + poss_ans1,
        "Sub-task 1: Synthesize and choose the most consistent parametrization and exponential forms.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: SC_CoT
    sc_instruction2 = "Sub-task 2: Convert the exponential expressions from Sub-task 1 into standard form a(theta)+b(theta)i and isolate the total real part as a function of cos(theta) and sin(theta)."
    N2 = self.max_sc
    sc_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    poss_think2, poss_ans2 = [], []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":sc_instruction2,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"SC_CoT"}
    for i in range(N2):
        thinking2_i, answer2_i = await sc_agents2[i]([taskInfo, thinking1, answer1], sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents2[i].id}, converting to a(theta)+b(theta)i, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        poss_think2.append(thinking2_i)
        poss_ans2.append(answer2_i)
    final_decision2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1] + poss_think2 + poss_ans2,
        "Sub-task 2: Synthesize and choose the most consistent a(theta)+b(theta)i and real-part expression.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: SC_CoT
    sc_instruction3 = "Sub-task 3: Combine the real-part expression from Sub-task 2 into the form A*cos(theta) + B*sin(theta), identifying the coefficients A and B explicitly."
    N3 = self.max_sc
    sc_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    poss_think3, poss_ans3 = [], []
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":sc_instruction3,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"SC_CoT"}
    for i in range(N3):
        thinking3_i, answer3_i = await sc_agents3[i]([taskInfo, thinking2, answer2], sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents3[i].id}, identifying A and B, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        poss_think3.append(thinking3_i)
        poss_ans3.append(answer3_i)
    final_decision3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision3([taskInfo, thinking2, answer2] + poss_think3 + poss_ans3,
        "Sub-task 3: Synthesize and choose the most consistent coefficients A and B.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: CoT
    cot_instruction4 = "Sub-task 4: Compute the amplitude R = sqrt(A^2 + B^2) to determine the maximum value of A*cos(theta) + B*sin(theta) over theta, and hence the maximal real part."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, computing amplitude R, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_instruction4,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"CoT"}
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Reflexion
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    reflect_instruction5 = "Sub-task 5: State the final maximal real part numerically and perform a brief verification at the maximizing theta. " + reflect_inst
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    inputs5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":reflect_instruction5,"context":["user query","thinking of subtask 4","answer of subtask 4"],"agent_collaboration":"Reflexion"}
    thinking5, answer5 = await cot_agent5(inputs5, reflect_instruction5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, initial final calculation, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max):
        feedback_i, correct_i = await critic_agent5([taskInfo, thinking5, answer5],
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, iteration {i}, feedback: {feedback_i.content}; correct: {correct_i.content}")
        if correct_i.content == "True":
            break
        inputs5.extend([thinking5, answer5, feedback_i])
        thinking5, answer5 = await cot_agent5(inputs5, reflect_instruction5, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refinement {i+1}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs