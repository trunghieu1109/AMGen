async def forward_101(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_agents_sc = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    
    # Stage 1: Analyze molecular and structural features, vector design
    cot_instruction_1 = "Sub-task 1: Analyze the molecular and structural features of the ligand and receptor proteins, including their domain organization, fusion tags (mCherry and eGFP), and regulatory elements (promoters, IRES, loxP and lox2272 stop cassettes) in the bicistronic vector construct to understand expected expression and fluorescence patterns."
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing molecular and structural features, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_instruction_2 = "Sub-task 2: Interpret the design and function of the bicistronic lox-Cre vector, focusing on the arrangement and interaction of ORFs, promoters, IRES, and loxP/lox2272 stop cassettes, and predict the expected expression outcomes before Cre recombination and after Cre-mediated excision."
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_agents_sc[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, interpreting vector design, thinking: {thinking2.content}; answer: {answer2.content}")
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
    
    # Stage 2: Evaluate Western blot results and Cre recombination efficiency
    cot_instruction_3a = "Sub-task 3a: Evaluate Western blot results for the ligand-mCherry fusion protein in transfected primary astrocyte cultures: confirm presence, correct molecular weight, and expression level relative to actin loading control."
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3a = self.max_round
    cot_inputs_3a = [taskInfo, thinking2, answer2]
    thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_instruction_3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, evaluating ligand Western blot, thinking: {thinking3a.content}; answer: {answer3a.content}")
    for i in range(N_max_3a):
        feedback, correct = await critic_agent_3a([taskInfo, thinking3a, answer3a], "please review the ligand Western blot evaluation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3a.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3a.extend([thinking3a, answer3a, feedback])
        thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_instruction_3a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, refining ligand Western blot evaluation, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_instruction_3b = "Sub-task 3b: Evaluate Western blot results for the receptor-eGFP fusion protein in transfected primary astrocyte cultures: confirm presence, correct molecular weight, and expression level relative to actin loading control to verify in-frame fusion and expression."
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3b = self.max_round
    cot_inputs_3b = [taskInfo, thinking2, answer2]
    thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, evaluating receptor Western blot, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_max_3b):
        feedback, correct = await critic_agent_3b([taskInfo, thinking3b, answer3b], "please review the receptor Western blot evaluation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3b.extend([thinking3b, answer3b, feedback])
        thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining receptor Western blot evaluation, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_instruction_3c = "Sub-task 3c: Assess Cre recombination efficiency and specificity in the SOX10-Cre hemizygous mouse line, including mapping SOX10-Cre expression domain and overlap with receptor-expressing cells to evaluate whether recombination should activate receptor-eGFP expression in neural crest derivatives."
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_instruction_3c,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3c = self.max_round
    cot_inputs_3c = [taskInfo, thinking2, answer2]
    thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_instruction_3c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, assessing Cre recombination efficiency and domain, thinking: {thinking3c.content}; answer: {answer3c.content}")
    for i in range(N_max_3c):
        feedback, correct = await critic_agent_3c([taskInfo, thinking3c, answer3c], "please review the Cre recombination efficiency and domain assessment and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3c.extend([thinking3c, answer3c, feedback])
        thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_instruction_3c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, refining Cre recombination assessment, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])
    
    # Stage 3: Analyze spatial overlap and protein localization
    cot_instruction_4a = "Sub-task 4a: Analyze spatial and cellular overlap between SOX10-Cre expression domain and receptor-expressing cells to determine if ligand-receptor interaction is autocrine or paracrine, and whether Cre recombination should lead to eGFP expression in the observed cells."
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4a = self.max_round
    cot_inputs_4a = [taskInfo, thinking3c, answer3c]
    thinking4a, answer4a = await cot_agent_4a(cot_inputs_4a, cot_instruction_4a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, analyzing spatial overlap and signaling mode, thinking: {thinking4a.content}; answer: {answer4a.content}")
    for i in range(N_max_4a):
        feedback, correct = await critic_agent_4a([taskInfo, thinking4a, answer4a], "please review the spatial overlap analysis and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4a.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4a.extend([thinking4a, answer4a, feedback])
        thinking4a, answer4a = await cot_agent_4a(cot_inputs_4a, cot_instruction_4a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, refining spatial overlap analysis, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    
    cot_instruction_4b = "Sub-task 4b: Evaluate protein localization of receptor-eGFP fusion in cells (e.g., via immunofluorescence or subcellular fractionation) to rule out mislocalization such as retention in the Golgi apparatus that could prevent fluorescence detection."
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_instruction_4b,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4b = self.max_round
    cot_inputs_4b = [taskInfo, thinking3b, answer3b]
    thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, cot_instruction_4b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, evaluating receptor-eGFP protein localization, thinking: {thinking4b.content}; answer: {answer4b.content}")
    for i in range(N_max_4b):
        feedback, correct = await critic_agent_4b([taskInfo, thinking4b, answer4b], "please review the protein localization evaluation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4b.extend([thinking4b, answer4b, feedback])
        thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, cot_instruction_4b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, refining protein localization evaluation, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    
    # Stage 4: Integrate data and evaluate hypotheses with Debate
    debate_instruction_5 = "Sub-task 5: Integrate all experimental data (Western blot results for ligand and receptor, Cre recombination efficiency and domain, protein localization, and fluorescence microscopy observations) to systematically generate and evaluate all plausible hypotheses explaining the absence of green fluorescence signal."
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 3b", "answer of subtask 3b", "thinking of subtask 3c", "answer of subtask 3c", "thinking of subtask 4a", "answer of subtask 4a", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "Debate"
    }
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking5, answer5 = await agent([
                    taskInfo, thinking3a, answer3a, thinking3b, answer3b, thinking3c, answer3c, thinking4a, answer4a, thinking4b, answer4b
                ], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [
                    taskInfo, thinking3a, answer3a, thinking3b, answer3b, thinking3c, answer3c, thinking4a, answer4a, thinking4b, answer4b
                ] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating data and evaluating hypotheses, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([
        taskInfo
    ] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the most likely reason for the absence of green fluorescence signal based on integrated evidence.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding on absence of green fluorescence cause, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    debate_instruction_6 = "Sub-task 6: Compare the multiple-choice answer options against the integrated analysis and evidence from previous subtasks to identify the most likely explanation for the lack of green fluorescence in the offspring."
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Debate"
    }
    all_thinking6 = [[] for _ in range(N_max_5)]
    all_answer6 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing answer options, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([
        taskInfo
    ] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final selection of the most likely explanation for the lack of green fluorescence.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting most likely explanation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
