async def forward_173(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1, Sub-task 1: Determine m1 and m2 (SC_CoT)
    cot_sc_instruction = (
        "Sub-task 1: Determine the rest masses m1 and m2 of the fission fragments in GeV. "
        "Use m1=(2/3)*0.99*M and m2=(1/3)*0.99*M with M*c^2=300 GeV. Ensure unit consistency."
    )
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                   for _ in range(self.max_sc)]
    possible_thk1, possible_ans1 = [], []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction,
                     "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents1:
        thinking1, answer1 = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thk1.append(thinking1)
        possible_ans1.append(answer1)
    final_decision1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1(
        [taskInfo] + possible_thk1 + possible_ans1,
        "Sub-task 1: Synthesize and choose the most consistent m1 and m2.", is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1, Sub-task 2: Compute Q = 0.01*M*c^2 = 3 GeV (SC_CoT)
    cot_sc_instruction2 = (
        "Sub-task 2: Compute the total kinetic-energy release Q = M*c^2 - (m1*c^2 + m2*c^2). "
        "Verify Q equals 0.01*M*c^2 = 3 GeV using m1 and m2 from Sub-task 1."
    )
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                   for _ in range(self.max_sc)]
    possible_thk2, possible_ans2 = [], []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction2,
                     "context": ["user query", thinking1, answer1], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents2:
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thk2.append(thinking2)
        possible_ans2.append(answer2)
    final_decision2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2(
        [taskInfo, thinking1, answer1] + possible_thk2 + possible_ans2,
        "Sub-task 2: Synthesize and confirm Q = 3 GeV.", is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2, Sub-task 3: Solve for p via Q = sum relativistic T (SC_CoT)
    cot_sc_instruction3 = (
        "Sub-task 3: Solve numerically for the common momentum p by inverting Q = [√(m1^2 + p^2) - m1] + [√(m2^2 + p^2) - m2] without approximation."
    )
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                   for _ in range(self.max_sc)]
    possible_thk3, possible_ans3 = [], []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction3,
                     "context": ["user query", thinking1, answer1, thinking2, answer2], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents3:
        thinking3, answer3 = await agent([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_thk3.append(thinking3)
        possible_ans3.append(answer3)
    final_decision3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision3(
        [taskInfo, thinking1, answer1, thinking2, answer2] + possible_thk3 + possible_ans3,
        "Sub-task 3: Synthesize and choose the most consistent momentum p.", is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 3, Sub-task 4: Compute T1_rel = √(m1^2 + p^2) - m1 (SC_CoT)
    cot_sc_instruction4 = (
        "Sub-task 4: Compute the heavier fragment's relativistic kinetic energy T1_rel = √(m1^2 + p^2) - m1 with high precision."
    )
    cot_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                   for _ in range(self.max_sc)]
    possible_thk4, possible_ans4 = [], []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction4,
                     "context": ["user query", thinking1, answer1, thinking3, answer3], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents4:
        thinking4, answer4 = await agent([taskInfo, thinking1, answer1, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_thk4.append(thinking4)
        possible_ans4.append(answer4)
    final_decision4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision4(
        [taskInfo, thinking1, answer1, thinking3, answer3] + possible_thk4 + possible_ans4,
        "Sub-task 4: Synthesize and choose the most consistent T1_rel.", is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 3, Sub-task 5: Assess classical approximation validity (Reflexion)
    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong. "
        "Using insights, assess if v1/c is sufficiently small or if higher-order terms are needed."
    )
    cot_reflect_instruction5 = f"Sub-task 5: Assess the validity of the classical KE approximation.{reflect_inst}"
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_reflect_instruction5,
                     "context": ["user query", thinking3, answer3], "agent_collaboration": "Reflexion"}
    inputs5 = [taskInfo, thinking3, answer3]
    thinking5, answer5 = await cot_agent5(inputs5, cot_reflect_instruction5, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    for _ in range(self.max_round):
        feedback5, correct5 = await critic_agent5([taskInfo, thinking5, answer5],
            "Please review and criticize where the above solution might be wrong. If correct, output exactly 'True' in 'correct'.", is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback: {feedback5.content}; correct: {correct5.content}")
        if correct5.content.strip() == "True":
            break
        inputs5 += [thinking5, answer5, feedback5]
        thinking5, answer5 = await cot_agent5(inputs5, cot_reflect_instruction5, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refinement thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 3, Sub-task 6: Compute T1_cl (and correction) (SC_CoT)
    cot_sc_instruction6 = (
        "Sub-task 6: Compute the classical KE T1_cl = p^2/(2*m1). If v1/c is not small, also include the next term -p^4/(8*m1^3)."   
    )
    cot_agents6 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                   for _ in range(self.max_sc)]
    possible_thk6, possible_ans6 = [], []
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_sc_instruction6,
                     "context": ["user query", thinking3, answer3, thinking5, answer5], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents6:
        thinking6, answer6 = await agent([taskInfo, thinking3, answer3, thinking5, answer5], cot_sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_thk6.append(thinking6)
        possible_ans6.append(answer6)
    final_decision6 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision6(
        [taskInfo, thinking3, answer3, thinking5, answer5] + possible_thk6 + possible_ans6,
        "Sub-task 6: Synthesize and choose the most consistent classical KE (with correction if needed).", is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    # Stage 4, Sub-task 7: Compute ΔT and select from options (Debate)
    debate_instr = (
        "Given solutions from other agents, consider their opinions. Please think carefully and provide an updated answer."
    )
    debate_instruction7 = (
        "Sub-task 7: Calculate ΔT = T1_rel - T1_cl and match to the choices {2,5,10,20} MeV." + debate_instr
    )
    debate_agents7 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                      for role in self.debate_role]
    all_thk7 = [[] for _ in range(self.max_round)]
    all_ans7 = [[] for _ in range(self.max_round)]
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": debate_instruction7,
                     "context": ["user query", thinking4, answer4, thinking6, answer6], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents7:
            if r == 0:
                thk7, ans7 = await agent([taskInfo, thinking4, answer4, thinking6, answer6], debate_instruction7, r, is_sub_task=True)
            else:
                inputs7 = [taskInfo, thinking4, answer4, thinking6, answer6] + all_thk7[r-1] + all_ans7[r-1]
                thk7, ans7 = await agent(inputs7, debate_instruction7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thk7.content}; answer: {ans7.content}")
            all_thk7[r].append(thk7)
            all_ans7[r].append(ans7)
    final_decision7 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision7(
        [taskInfo, thinking4, answer4, thinking6, answer6] + all_thk7[-1] + all_ans7[-1],
        "Sub-task 7: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs