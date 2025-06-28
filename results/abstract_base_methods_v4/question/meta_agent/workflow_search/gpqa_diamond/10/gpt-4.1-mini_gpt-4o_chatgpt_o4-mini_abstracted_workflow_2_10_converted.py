async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    N_sc = self.max_sc
    cot_agents_sc = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    cot_agent_reflexion = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_reflexion = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)

    cot_sc_instruction_1 = "Sub-task 1: Analyze the molecular biology concepts related to programmed ribosomal frameshifting in SARS-CoV-2, including the mechanism involving slippery nucleotides and pseudoknots, and compare the conformation of SARS-CoV-2 frameshifting signals with those of SARS-CoV."
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    for i in range(N_sc):
        thinking1, answer1 = await cot_agents_sc[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, attempt {i}, analyzing programmed ribosomal frameshifting, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1.content)
        thinkingmapping_1[answer1.content] = thinking1
        answermapping_1[answer1.content] = answer1
    answer1_content = Counter(possible_answers_1).most_common(1)[0][0]
    thinking1 = thinkingmapping_1[answer1_content]
    answer1 = answermapping_1[answer1_content]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Examine the role and molecular function of the SARS-CoV-2 nsp10/nsp14-ExoN complex, focusing on its heterodimer formation, mismatch repair activity, and the interaction between nsp14's ExoN domain and nsp10 in preventing dsRNA breakdown."
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_sc[i]([taskInfo], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, attempt {i}, examining nsp10/nsp14-ExoN complex, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = "Sub-task 3: Investigate the apoptotic pathways triggered by SARS-CoV-2 ORF3a, specifically its ability to activate caspase-8 without altering Bcl-2 expression, and clarify the distinction between extrinsic and mitochondrial apoptotic pathways in this context."
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_sc[i]([taskInfo], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, attempt {i}, investigating ORF3a apoptotic pathways, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4a = "Sub-task 4a: Analyze the structural conformations of pseudoknots in SARS-CoV and SARS-CoV-2 under tension, identifying the number and nature of conformations experimentally observed."
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_sc_instruction_4a,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_4a = []
    thinkingmapping_4a = {}
    answermapping_4a = {}
    for i in range(N_sc):
        thinking4a, answer4a = await cot_agents_sc[i]([taskInfo], cot_sc_instruction_4a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, attempt {i}, analyzing pseudoknot conformations under tension, thinking: {thinking4a.content}; answer: {answer4a.content}")
        possible_answers_4a.append(answer4a.content)
        thinkingmapping_4a[answer4a.content] = thinking4a
        answermapping_4a[answer4a.content] = answer4a
    answer4a_content = Counter(possible_answers_4a).most_common(1)[0][0]
    thinking4a = thinkingmapping_4a[answer4a_content]
    answer4a = answermapping_4a[answer4a_content]
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])

    cot_sc_instruction_4b = "Sub-task 4b: Examine the factors influencing programmed -1 ribosomal frameshifting rates in SARS-CoV and SARS-CoV-2, including but not limited to pseudoknot conformations, and gather quantitative data and literature evidence on frameshifting efficiency."
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_sc_instruction_4b,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_4b = []
    thinkingmapping_4b = {}
    answermapping_4b = {}
    for i in range(N_sc):
        thinking4b, answer4b = await cot_agents_sc[i]([taskInfo], cot_sc_instruction_4b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, attempt {i}, examining frameshifting rates and influencing factors, thinking: {thinking4b.content}; answer: {answer4b.content}")
        possible_answers_4b.append(answer4b.content)
        thinkingmapping_4b[answer4b.content] = thinking4b
        answermapping_4b[answer4b.content] = answer4b
    answer4b_content = Counter(possible_answers_4b).most_common(1)[0][0]
    thinking4b = thinkingmapping_4b[answer4b_content]
    answer4b = answermapping_4b[answer4b_content]
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])

    cot_reflect_instruction_4c = "Sub-task 4c: Critically evaluate the claim that the rate of frameshifting is linearly correlated with the number of pseudoknot conformations by integrating findings from subtasks 4a and 4b, considering alternative explanations and multifactorial influences, supported by key literature citations such as Ritchie et al., Nucleic Acids Research 2020."
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_reflect_instruction_4c,
        "context": ["user query", thinking4a, answer4a, thinking4b, answer4b],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_4c = [taskInfo, thinking4a, answer4a, thinking4b, answer4b]
    thinking4c, answer4c = await cot_agent_reflexion(cot_inputs_4c, cot_reflect_instruction_4c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, initial critical evaluation of frameshifting rate correlation, thinking: {thinking4c.content}; answer: {answer4c.content}")
    N_max_reflexion = self.max_round
    for i in range(N_max_reflexion):
        feedback, correct = await critic_agent_reflexion([taskInfo, thinking4c, answer4c],
                                                       "Critically assess the validity and limitations of the correlation claim between frameshifting rate and pseudoknot conformations, including alternative explanations and empirical evidence.",
                                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_reflexion.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4c.extend([thinking4c, answer4c, feedback])
        thinking4c, answer4c = await cot_agent_reflexion(cot_inputs_4c, cot_reflect_instruction_4c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, refining critical evaluation, thinking: {thinking4c.content}; answer: {answer4c.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {"thinking": thinking4c, "answer": answer4c}
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])

    debate_instruction_5 = "Sub-task 5: Compare and contrast the findings from subtasks 1, 2, 3, and 4c to identify inconsistencies or inaccuracies in the given statements about SARS-CoV-2 molecular biology, focusing on the correctness of each choice."
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4c, answer4c],
        "agent_collaboration": "Debate"
    }
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4c, answer4c], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4c, answer4c] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing subtasks 1-4c, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 6: Determine which statement among the provided choices is incorrect based on the integrated analysis from subtask 5, and select the corresponding letter (A, B, C, or D) as the final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining incorrect statement, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": "Determine the incorrect statement based on integrated analysis.",
        "context": ["user query", thinking5, answer5],
        "agent_collaboration": "Final Decision"
    }
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
