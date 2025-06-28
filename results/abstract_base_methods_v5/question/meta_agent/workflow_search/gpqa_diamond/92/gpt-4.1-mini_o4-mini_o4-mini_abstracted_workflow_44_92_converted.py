async def forward_92(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Extract and organize the qPCR experimental data from the query, including concentrations, triplicate Ct values, and given parameters (efficiency, R2, slope), to establish a clear dataset for analysis."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extract and organize qPCR data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Calculate the mean Ct values for each concentration from the triplicate Ct values to summarize the data for subsequent slope and variability analyses, based on the extracted data from Sub-task 1."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, calculate mean Ct values, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_instruction_3a = "Sub-task 3a: Calculate the observed calibration slopes (ΔCt/Δlog10 concentration) between all adjacent dilution points using the mean Ct values, explicitly noting the sign and magnitude of each slope to verify the expected negative slope."
    cot_sc_instruction_3b = "Sub-task 3b: Compare the observed slopes and their signs from Sub-task 3a to the expected slope of -3.3 cycles per ten-fold dilution, checking for consistency with the biological principle that Ct increases as concentration decreases; flag any inconsistencies or reversals."
    cot_sc_instruction_3c = "Sub-task 3c: Cross-validate the Ct versus concentration relationship by confirming that Ct values increase with decreasing concentration across all dilution steps, ensuring logical consistency before proceeding to replicate variability assessment."
    N = self.max_sc
    cot_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3a = []
    thinkingmapping_3a = {}
    answermapping_3a = {}
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3a, answer3a = await cot_agents_3a[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, calculate observed calibration slopes, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a.content)
        thinkingmapping_3a[answer3a.content] = thinking3a
        answermapping_3a[answer3a.content] = answer3a
    answer3a_content = Counter(possible_answers_3a).most_common(1)[0][0]
    thinking3a = thinkingmapping_3a[answer3a_content]
    answer3a = answermapping_3a[answer3a_content]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    cot_reflect_instruction_3b = cot_sc_instruction_3b
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_3b = [taskInfo, thinking3a, answer3a]
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "Reflexion"
    }
    thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, compare observed slopes to expected slope, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_3b([taskInfo, thinking3b, answer3b], "please review the slope comparison and flag any inconsistencies or reversals.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3b.extend([thinking3b, answer3b, feedback])
        thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining slope comparison, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    cot_reflect_instruction_3c = cot_sc_instruction_3c
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_3c = [taskInfo, thinking2, answer2, thinking3b, answer3b]
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_reflect_instruction_3c,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Reflexion"
    }
    thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, cross-validate Ct vs concentration relationship, thinking: {thinking3c.content}; answer: {answer3c.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_3c([taskInfo, thinking3c, answer3c], "please review the Ct versus concentration relationship validation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3c.extend([thinking3c, answer3c, feedback])
        thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, refining Ct vs concentration validation, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {
        "thinking": thinking3c,
        "answer": answer3c
    }
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])
    cot_sc_instruction_4a = "Sub-task 4a: Calculate multiple statistical measures of replicate variability for each concentration, including range, standard deviation, and coefficient of variation (CV), based on the triplicate Ct values extracted in Sub-task 1."
    N = self.max_sc
    cot_agents_4a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4a = []
    thinkingmapping_4a = {}
    answermapping_4a = {}
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_sc_instruction_4a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4a, answer4a = await cot_agents_4a[i]([taskInfo, thinking1, answer1], cot_sc_instruction_4a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4a[i].id}, calculate replicate variability metrics, thinking: {thinking4a.content}; answer: {answer4a.content}")
        possible_answers_4a.append(answer4a.content)
        thinkingmapping_4a[answer4a.content] = thinking4a
        answermapping_4a[answer4a.content] = answer4a
    answer4a_content = Counter(possible_answers_4a).most_common(1)[0][0]
    thinking4a = thinkingmapping_4a[answer4a_content]
    answer4a = answermapping_4a[answer4a_content]
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {
        "thinking": thinking4a,
        "answer": answer4a
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    cot_reflect_instruction_4b = "Sub-task 4b: Interpret the replicate variability metrics from Sub-task 4a in the context of qPCR experimental standards, considering expected variability thresholds and natural variability at low copy numbers to assess replicate consistency accurately."
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4b = [taskInfo, thinking4a, answer4a]
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_reflect_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "Reflexion"
    }
    thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, cot_reflect_instruction_4b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, interpret replicate variability metrics, thinking: {thinking4b.content}; answer: {answer4b.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_4b([taskInfo, thinking4b, answer4b], "please review the replicate variability interpretation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4b.extend([thinking4b, answer4b, feedback])
        thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, cot_reflect_instruction_4b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, refining replicate variability interpretation, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    debate_instruction_5 = "Sub-task 5: Integrate and interpret the findings from slope verification (Sub-tasks 3a-3c) and replicate variability assessment (Sub-tasks 4a-4b) to determine which provided choice best explains the discrepancies in the qPCR results."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3c, answer3c, thinking4b, answer4b], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3c, answer3c, thinking4b, answer4b] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating slope and variability findings, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the explanation for discrepancies in qPCR results.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision on explanation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_reflect_instruction_6 = "Sub-task 6: Perform a final validation by cross-checking all assumptions, statistical methods, and biological principles used in previous subtasks to ensure consistency and correctness before finalizing the answer."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_6 = [taskInfo, thinking5, answer5, thinking3a, answer3a, thinking3b, answer3b, thinking3c, answer3c, thinking4a, answer4a, thinking4b, answer4b]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "thinking and answer of subtask 5", "thinking and answers of subtasks 3a-3c", "thinking and answers of subtasks 4a-4b"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, final validation of all analyses, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "please review the final validation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining final validation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
