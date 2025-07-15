async def forward_153(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    N_sc = self.max_sc
    # Stage 0, Sub-task 1: SC-CoT
    instr1 = "Sub-task 1: Extract and summarize the mass spectrometry, IR, and ¹H NMR data from the query to prepare for analysis."
    sc_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    thinkings1 = []
    answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":instr1,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i, agent in enumerate(sc_agents1):
        thinking_i, answer_i = await agent([taskInfo], instr1, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, extracting data, thinking: {thinking_i.content}; answer: {answer_i.content}")
        thinkings1.append(thinking_i)
        answers1.append(answer_i)
    final_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = "Sub-task 1: Synthesize and choose the most consistent summary of the spectral data."
    thinking1, answer1 = await final_agent1([taskInfo] + thinkings1 + answers1, final_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 0, Sub-task 2: SC-CoT
    instr2 = "Sub-task 2: Based on the output from Sub-task 1, analyze the mass spec isotopic pattern (M and M+2 peaks) and IR absorptions to infer the presence of chlorine and key functional groups."
    sc_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    thinkings2 = []
    answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":instr2,"context":["user query","thinking1","answer1"],"agent_collaboration":"SC_CoT"}
    for agent in sc_agents2:
        thinking_i, answer_i = await agent([taskInfo, thinking1, answer1], instr2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, analyzing MS and IR, thinking: {thinking_i.content}; answer: {answer_i.content}")
        thinkings2.append(thinking_i)
        answers2.append(answer_i)
    final_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Sub-task 2: Synthesize and choose the most consistent inference for isotopic pattern and IR."
    thinking2, answer2 = await final_agent2([taskInfo, thinking1, answer1] + thinkings2 + answers2, final_instr2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 0, Sub-task 3: SC-CoT
    instr3 = "Sub-task 3: Based on the output from Sub-task 1, interpret the ¹H NMR chemical shifts, integration, and splitting to deduce the proton environments and aromatic substitution pattern."
    sc_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    thinkings3 = []
    answers3 = []
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":instr3,"context":["user query","thinking1","answer1"],"agent_collaboration":"SC_CoT"}
    for agent in sc_agents3:
        thinking_i, answer_i = await agent([taskInfo, thinking1, answer1], instr3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, interpreting NMR, thinking: {thinking_i.content}; answer: {answer_i.content}")
        thinkings3.append(thinking_i)
        answers3.append(answer_i)
    final_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Sub-task 3: Synthesize and choose the most consistent NMR interpretation."
    thinking3, answer3 = await final_agent3([taskInfo, thinking1, answer1] + thinkings3 + answers3, final_instr3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Stage 1, Sub-task 4: Debate
    instr4 = "Sub-task 4: Compare the inferred functional groups and substitution symmetry against each candidate structure to narrow down plausible options. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    R = self.max_round
    all_think4 = [[] for _ in range(R)]
    all_ans4 = [[] for _ in range(R)]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":instr4,"context":["user query","thinking2","answer2","thinking3","answer3"],"agent_collaboration":"Debate"}
    for r in range(R):
        for agent in debate_agents:
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking2, answer2, thinking3, answer3], instr4, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2, thinking3, answer3] + all_think4[r-1] + all_ans4[r-1]
                thinking_i, answer_i = await agent(inputs, instr4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_think4[r].append(thinking_i)
            all_ans4[r].append(answer_i)
    final_agent4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_agent4([taskInfo, thinking2, answer2, thinking3, answer3] + all_think4[-1] + all_ans4[-1], "Sub-task 4: " + final_instr4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    # Stage 2, Sub-task 5: Reflexion
    reflect = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    instr5 = "Sub-task 5: Integrate all spectroscopic evidence and select the most reasonable structure from the provided choices. " + reflect
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs5 = [taskInfo, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":instr5,"context":["user query","thinking2","answer2","thinking3","answer3","thinking4","answer4"],"agent_collaboration":"Reflexion"}
    thinking5, answer5 = await cot_agent(inputs5, instr5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, initial integration, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(self.max_round):
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback_i, correct_i = await critic([taskInfo, thinking5, answer5], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic.id}, feedback: {feedback_i.content}; correct: {correct_i.content}")
        if correct_i.content == "True":
            break
        inputs5 += [thinking5, answer5, feedback_i]
        thinking5, answer5 = await cot_agent(inputs5, instr5, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs