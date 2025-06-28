async def forward_92(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    cot_instruction_1 = "Sub-task 1: Extract and organize the qPCR experimental data including concentrations, Ct values for triplicates, and calibration curve parameters (efficiency, R2, slope) from the query."
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting and organizing qPCR data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Calculate the mean Ct values for each concentration from the triplicate Ct values to summarize the data for further analysis."
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_sc_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, calculating mean Ct values, thinking: {thinking2.content}; answer: {answer2.content}")
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
    cot_reflect_instruction_3a = "Sub-task 3a: Verify the fundamental qPCR principle by analyzing the directionality of the relationship between mean Ct values and known concentrations: confirm that Ct values decrease as concentration increases, using tabulation or visualization, and explicitly report if this expected inverse relationship holds or if inconsistencies exist. Include original Ct values and concentrations in your analysis."
    cot_inputs_3a = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_reflect_instruction_3a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3a, answer3a = await cot_agent(cot_inputs_3a, cot_reflect_instruction_3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, verifying Ct-concentration directionality, thinking: {thinking3a.content}; answer: {answer3a.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3a, answer3a], "please review the Ct-concentration directionality verification and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3a.extend([thinking3a, answer3a, feedback])
        thinking3a, answer3a = await cot_agent(cot_inputs_3a, cot_reflect_instruction_3a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining Ct-concentration directionality verification, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    cot_reflect_instruction_3b = "Sub-task 3b: Assess the magnitude of Ct differences between each ten-fold serial dilution by comparing observed mean Ct differences to the expected value (~3.3 cycles per ten-fold dilution) based on the slope, and identify any deviations from expected qPCR efficiency. Use the mean Ct values from Sub-task 2 and the calibration curve slope from Sub-task 1."
    cot_inputs_3b = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3a, answer3a]
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "Reflexion"
    }
    thinking3b, answer3b = await cot_agent(cot_inputs_3b, cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, assessing Ct difference magnitude, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3b, answer3b], "please review the assessment of Ct differences between dilutions and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3b.extend([thinking3b, answer3b, feedback])
        thinking3b, answer3b = await cot_agent(cot_inputs_3b, cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining Ct difference magnitude assessment, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    cot_instruction_5 = "Sub-task 4: Evaluate the consistency of technical replicates by calculating the deviation among triplicate Ct values for each concentration and comparing it to an acceptable threshold (0.3 cycles) to identify replicate variability. Use the original Ct values from Sub-task 1."
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, evaluating replicate consistency, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_reflect_instruction_6 = "Sub-task 5: Interpret the calibration curve parameters (efficiency, R2, slope) in the context of the data and verify if the calibration curve is valid and if qPCR is suitable for quantification in this experiment. Use outputs from Sub-tasks 1, 3a, and 3b."
    cot_inputs_6 = [taskInfo, thinking1, answer1, thinking3a, answer3a, thinking3b, answer3b]
    subtask_desc6 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, interpreting qPCR parameters, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking6, answer6], "please review the interpretation of qPCR efficiency, R2, and slope and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining interpretation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 5: ", sub_tasks[-1])
    debate_instruction_7 = "Sub-task 6: Integrate findings from the Ct trend verification (Sub-task 3a), dilution effect assessment (Sub-task 3b), replicate deviation analysis (Sub-task 4), and calibration curve evaluation (Sub-task 5) to determine which explanation among the provided choices best accounts for the observed discrepancies, ensuring that fundamental qPCR principles are upheld and any inconsistencies are clearly identified."
    subtask_desc7 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_7,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 3b", "answer of subtask 3b", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Debate"
    }
    all_thinking7 = [[] for _ in range(N_max)]
    all_answer7 = [[] for _ in range(N_max)]
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking3a, answer3a, thinking3b, answer3b, thinking4, answer4, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking3a, answer3a, thinking3b, answer3b, thinking4, answer4, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating findings and deciding explanation, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 6: Make final decision on the best explanation for the observed discrepancies.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs