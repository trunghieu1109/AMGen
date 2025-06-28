async def forward_90(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1a = "Sub-task 1a: Determine the detailed structure of product 1 formed when cyclohexanone is treated with LDA at low temperature, followed by benzaldehyde addition and acidification. Include identification of the aldol addition product, assign absolute and relative stereochemistry, analyze chair conformations, and specify stereochemical descriptors (R/S) for all stereocenters formed. Use stereochemical visualization or standardized notation to reduce ambiguity."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1a, answer1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, detailed stereochemical determination of product 1, thinking: {thinking1a.content}; answer: {answer1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc1a['response'] = {
        "thinking": thinking1a,
        "answer": answer1a
    }
    logs.append(subtask_desc1a)
    print("Step 1a: ", sub_tasks[-1])
    cot_instruction_1b = "Sub-task 1b: Explicitly confirm and rationalize the stereochemical outcome of the aldol addition step leading to product 1. Discuss the stereochemical relationship between the newly formed stereocenters, favored chair conformation, and stereochemical propagation to ensure accuracy."
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "CoT"
    }
    thinking1b, answer1b = await cot_agent_1b([taskInfo, thinking1a, answer1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, stereochemical confirmation of aldol addition, thinking: {thinking1b.content}; answer: {answer1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc1b['response'] = {
        "thinking": thinking1b,
        "answer": answer1b
    }
    logs.append(subtask_desc1b)
    print("Step 1b: ", sub_tasks[-1])
    cot_sc_instruction_2a = "Sub-task 2a: Analyze the fate of the beta-hydroxy group in product 1 when treated with excess diethylaminosulfur trifluoride (DAST). Consider stereochemical inversion and mechanistic pathways."
    cot_sc_instruction_2b = "Sub-task 2b: Analyze the fate of the ketone group in product 1 under excess DAST treatment, including the possibility of geminal difluoride formation. Discuss mechanistic pathways and stereochemical implications."
    cot_sc_instruction_2c = "Sub-task 2c: Reconcile the predictions from 2a and 2b to enumerate all plausible product structures and their stereochemical configurations after DAST treatment."
    N = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_2c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking1b, answer1b], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, analyzing beta-hydroxy group fluorination, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
    answer2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[answer2a_content]
    answer2a = answermapping_2a[answer2a_content]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {
        "thinking": thinking2a,
        "answer": answer2a
    }
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking1b, answer1b], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, analyzing ketone group fluorination, thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b.content)
        thinkingmapping_2b[answer2b.content] = thinking2b
        answermapping_2b[answer2b.content] = answer2b
    answer2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking2b = thinkingmapping_2b[answer2b_content]
    answer2b = answermapping_2b[answer2b_content]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {
        "thinking": thinking2b,
        "answer": answer2b
    }
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])
    possible_answers_2c = []
    thinkingmapping_2c = {}
    answermapping_2c = {}
    subtask_desc2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_sc_instruction_2c,
        "context": ["user query", "thinking of subtask_2a", "answer of subtask_2a", "thinking of subtask_2b", "answer of subtask_2b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2c, answer2c = await cot_agents_2c[i]([taskInfo, thinking2a, answer2a, thinking2b, answer2b], cot_sc_instruction_2c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2c[i].id}, reconciling DAST product predictions, thinking: {thinking2c.content}; answer: {answer2c.content}")
        possible_answers_2c.append(answer2c.content)
        thinkingmapping_2c[answer2c.content] = thinking2c
        answermapping_2c[answer2c.content] = answer2c
    answer2c_content = Counter(possible_answers_2c).most_common(1)[0][0]
    thinking2c = thinkingmapping_2c[answer2c_content]
    answer2c = answermapping_2c[answer2c_content]
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking2c.content}; answer - {answer2c.content}")
    subtask_desc2c['response'] = {
        "thinking": thinking2c,
        "answer": answer2c
    }
    logs.append(subtask_desc2c)
    print("Step 2c: ", sub_tasks[-1])
    debate_instruction_3 = "Sub-task 3: Perform a detailed stereochemical analysis of the predicted products from DAST treatment, including absolute configurations and possible stereochemical outcomes for each fluorinated center. Compare mechanistic pathways and stereochemical consequences to identify the most chemically plausible product 2."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", "thinking of subtask_2c", "answer of subtask_2c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2c, answer2c], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2c, answer2c] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, stereochemical analysis and product plausibility, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make final decision on the most plausible product 2 structure after DAST treatment.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct product 2 structure, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_reflect_instruction_4 = "Sub-task 4: Conduct a final consistency check and reflection on the entire reasoning chain, verifying stereochemical assignments, mechanistic assumptions, and product predictions to ensure no contradictions or overlooked pathways remain before finalizing the answer."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking1a, answer1a, thinking1b, answer1b, thinking2a, answer2a, thinking2b, answer2b, thinking2c, answer2c, thinking3, answer3]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "all previous thinking and answers"],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, final consistency check, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "Please review the entire reasoning chain for stereochemical and mechanistic consistency and provide limitations or contradictions.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining final consistency check, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    debate_instruction_5 = "Sub-task 5: Critically compare the predicted structure(s) of product 2 with the given multiple-choice options. Evaluate molecular connectivity, functional groups, and stereochemical assignments to select the correct answer choice that matches the predicted product 2 structure."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing predicted product 2 with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct structure of product 2 among the multiple-choice options.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct product 2 choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_reflect_instruction_6 = "Sub-task 6: Conduct a final consistency check and reflection on the entire reasoning chain, verifying stereochemical assignments, mechanistic assumptions, and product predictions to ensure no contradictions or overlooked pathways remain before finalizing the answer."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking1a, answer1a, thinking1b, answer1b, thinking2a, answer2a, thinking2b, answer2b, thinking2c, answer2c, thinking3, answer3, thinking4, answer4, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "all previous thinking and answers"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, final overall consistency check, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "Please review the entire workflow reasoning and final answer for consistency and completeness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining final overall consistency check, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs