async def forward_167(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Extract and summarize the question's key elements, define what makes an error difficult-to-spot (silent vs overt)."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting key elements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Develop a structured scoring rubric classifying each error source by detectability (overt vs silent) and prevalence."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking1", "answer1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        t2, a2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, rubric draft, thinking: {t2.content}; answer: {a2.content}")
        possible_thinkings.append(t2)
        possible_answers.append(a2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Given all prior thinking and answers, synthesize and choose the most consistent rubric."
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + possible_thinkings + possible_answers,
        "Sub-task 2: Synthesize rubric. " + final_instr2,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr = "Given solutions from other agents, consider their opinions and provide an updated evaluation."
    debate_instruction_3 = "Sub-task 3: Apply the rubric to evaluate each error source and produce a table of scores. " + debate_instr
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinkings3 = [[] for _ in range(N_max_3)]
    all_answers3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": debate_instruction_3, "context": ["user query", "thinking2", "answer2"], "agent_collaboration": "Debate"}
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                t3, a3 = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2] + all_thinkings3[r-1] + all_answers3[r-1]
                t3, a3 = await agent(inputs, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t3.content}; answer: {a3.content}")
            all_thinkings3[r].append(t3)
            all_answers3[r].append(a3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Given all debate outputs, provide a final table of scores."
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo, thinking2, answer2] + all_thinkings3[-1] + all_answers3[-1],
        "Sub-task 3: Synthesize final scoring table. " + final_instr3,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction4 = "Sub-task 4: Map the scored table onto the multiple-choice options to identify subtle error sources."
    M = self.max_sc
    sc_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(M)]
    poss_think4 = []
    poss_ans4 = []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction4, "context": ["user query", "thinking3", "answer3"], "agent_collaboration": "SC_CoT"}
    for j in range(M):
        t4, a4 = await sc_agents4[j]([taskInfo, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents4[j].id}, mapping to choices, thinking: {t4.content}; answer: {a4.content}")
        poss_think4.append(t4)
        poss_ans4.append(a4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = "Given all the above thinking and answers, choose the choice(s) matching subtle error sources."
    thinking4, answer4 = await final_decision_agent_4(
        [taskInfo, thinking3, answer3] + poss_think4 + poss_ans4,
        "Sub-task 4: Synthesize mapping to choices. " + final_instr4,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong. Using insights, improve the mapping." 
    cot_reflect_instruction = "Sub-task 5: Reflexively re-examine the mapping and guard against confirmation bias." + reflect_inst
    cot_refl_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_reflect_instruction, "context": ["user query", "thinking4", "answer4"], "agent_collaboration": "Reflexion"}
    cot_inputs = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_refl_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_refl_agent.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(self.max_round):
        feedback5, correct5 = await critic_agent(
            [taskInfo, thinking5, answer5],
            "Please review where the answer might be wrong. If you are sure it is correct, output exactly 'True' in 'correct'",
            i,
            is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback5.content}; correct: {correct5.content}")
        if correct5.content == "True":
            break
        cot_inputs.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_refl_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_refl_agent.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs