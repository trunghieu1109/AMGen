async def forward_88(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_agents_1a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_instruction_1a = (
        "Sub-task 1a: Enumerate all plausible mechanistic pathways for the reaction of 1,3-dibromoadamantane heated with excess KOH at 240°C, "
        "including elimination and Favorskii rearrangement, to generate candidate structures for product 1."
    )
    subtask_desc1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_sc_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1a = []
    thinkingmapping_1a = {}
    answermapping_1a = {}
    for i in range(self.max_sc):
        thinking1a, answer1a = await cot_sc_agents_1a[i]([taskInfo], cot_sc_instruction_1a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1a[i].id}, enumerating mechanistic pathways, thinking: {thinking1a.content}; answer: {answer1a.content}")
        possible_answers_1a.append(answer1a.content)
        thinkingmapping_1a[answer1a.content] = thinking1a
        answermapping_1a[answer1a.content] = answer1a
    most_common_answer_1a = Counter(possible_answers_1a).most_common(1)[0][0]
    thinking1a = thinkingmapping_1a[most_common_answer_1a]
    answer1a = answermapping_1a[most_common_answer_1a]
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc1a)
    print("Step 1a: ", sub_tasks[-1])
    cot_sc_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_instruction_1b = (
        "Sub-task 1b: For each candidate structure of product 1 from Sub-task 1a, rigorously analyze and match the given 1H NMR (chemical shifts, integration, multiplicity) "
        "and IR data (notably the 1720 cm-1 absorbance) to confirm or exclude candidates, ensuring consistency with chemical logic and spectral evidence."
    )
    subtask_desc1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_sc_instruction_1b,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    for i in range(self.max_sc):
        thinking1b, answer1b = await cot_sc_agents_1b[i]([taskInfo, thinking1a, answer1a], cot_sc_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1b[i].id}, matching spectral data, thinking: {thinking1b.content}; answer: {answer1b.content}")
        possible_answers_1b.append(answer1b.content)
        thinkingmapping_1b[answer1b.content] = thinking1b
        answermapping_1b[answer1b.content] = answer1b
    most_common_answer_1b = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking1b = thinkingmapping_1b[most_common_answer_1b]
    answer1b = answermapping_1b[most_common_answer_1b]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc1b)
    print("Step 1b: ", sub_tasks[-1])
    cot_reflect_instruction_1c = (
        "Sub-task 1c: Select the most plausible structure of product 1 based on combined mechanistic reasoning and spectral data validation, "
        "and explicitly document the rationale."
    )
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1c = self.max_round
    cot_inputs_1c = [taskInfo, thinking1a, answer1a, thinking1b, answer1b]
    subtask_desc1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_reflect_instruction_1c,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "Reflexion"
    }
    thinking1c, answer1c = await cot_agent_1c(cot_inputs_1c, cot_reflect_instruction_1c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1c.id}, selecting plausible product 1, thinking: {thinking1c.content}; answer: {answer1c.content}")
    for i in range(N_max_1c):
        feedback, correct = await critic_agent_1c([taskInfo, thinking1c, answer1c], "please review the selection of product 1 structure and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1c.extend([thinking1c, answer1c, feedback])
        thinking1c, answer1c = await cot_agent_1c(cot_inputs_1c, cot_reflect_instruction_1c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1c.id}, refining product 1 selection, thinking: {thinking1c.content}; answer: {answer1c.content}")
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking1c.content}; answer - {answer1c.content}")
    subtask_desc1c['response'] = {"thinking": thinking1c, "answer": answer1c}
    logs.append(subtask_desc1c)
    print("Step 1c: ", sub_tasks[-1])
    cot_sc_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_instruction_2a = (
        "Sub-task 2a: Analyze the reaction of product 1 with excess aluminum isopropoxide under heating, considering all plausible reaction pathways "
        "(e.g., reduction, rearrangement, or other transformations) to propose candidate structures for product 2."
    )
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", "thinking of subtask 1c", "answer of subtask 1c"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    for i in range(self.max_sc):
        thinking2a, answer2a = await cot_sc_agents_2a[i]([taskInfo, thinking1c, answer1c], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2a[i].id}, proposing product 2 candidates, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
    most_common_answer_2a = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[most_common_answer_2a]
    answer2a = answermapping_2a[most_common_answer_2a]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    cot_reflect_instruction_2b = (
        "Sub-task 2b: Evaluate the plausibility of each candidate structure for product 2 by considering mechanistic feasibility and any available spectral or literature data, "
        "selecting the most consistent structure."
    )
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2b = self.max_round
    cot_inputs_2b = [taskInfo, thinking1c, answer1c, thinking2a, answer2a]
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_reflect_instruction_2b,
        "context": ["user query", "thinking of subtask 1c", "answer of subtask 1c", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "Reflexion"
    }
    thinking2b, answer2b = await cot_agent_2b(cot_inputs_2b, cot_reflect_instruction_2b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2b.id}, evaluating product 2 candidates, thinking: {thinking2b.content}; answer: {answer2b.content}")
    for i in range(N_max_2b):
        feedback, correct = await critic_agent_2b([taskInfo, thinking2b, answer2b], "please review the evaluation of product 2 candidates and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2b.extend([thinking2b, answer2b, feedback])
        thinking2b, answer2b = await cot_agent_2b(cot_inputs_2b, cot_reflect_instruction_2b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2b.id}, refining evaluation of product 2, thinking: {thinking2b.content}; answer: {answer2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])
    cot_sc_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_instruction_3a = (
        "Sub-task 3a: Analyze the ozonolysis reaction conditions applied to product 2 (ozone at -78°C followed by dimethyl sulfide workup), explicitly considering the presence or absence of unsaturation in product 2, "
        "to predict the structure(s) of product 3."
    )
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_3a = []
    thinkingmapping_3a = {}
    answermapping_3a = {}
    for i in range(self.max_sc):
        thinking3a, answer3a = await cot_sc_agents_3a[i]([taskInfo, thinking2b, answer2b], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3a[i].id}, predicting product 3 structure, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a.content)
        thinkingmapping_3a[answer3a.content] = thinking3a
        answermapping_3a[answer3a.content] = answer3a
    most_common_answer_3a = Counter(possible_answers_3a).most_common(1)[0][0]
    thinking3a = thinkingmapping_3a[most_common_answer_3a]
    answer3a = answermapping_3a[most_common_answer_3a]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    cot_reflect_instruction_3b = (
        "Sub-task 3b: Validate the predicted structure of product 3 by integrating mechanistic reasoning and any available spectral or chemical evidence, "
        "ensuring the product is chemically reasonable and consistent with ozonolysis chemistry."
    )
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3b = self.max_round
    cot_inputs_3b = [taskInfo, thinking2b, answer2b, thinking3a, answer3a]
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "Reflexion"
    }
    thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, validating product 3 structure, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_max_3b):
        feedback, correct = await critic_agent_3b([taskInfo, thinking3b, answer3b], "please review the validation of product 3 structure and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3b.extend([thinking3b, answer3b, feedback])
        thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining validation of product 3, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    debate_instruction_4 = (
        "Sub-task 4: Identify the most deshielded non-exchangeable proton in the 1H NMR spectrum of product 3, "
        "and analyze its coupling pattern by applying splitting rules and considering neighboring protons and their coupling constants."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3b, answer3b], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3b, answer3b] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing NMR coupling, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the coupling pattern of the most deshielded proton in product 3's 1H NMR spectrum.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding coupling pattern, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_sc_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_instruction_5 = (
        "Sub-task 5: Match the determined coupling pattern of the most deshielded proton in product 3's 1H NMR spectrum "
        "to the provided multiple-choice options (doublet of triplets, triplet of triplets, pentet, triplet) and select the correct answer (A, B, C, or D)."
    )
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    for i in range(self.max_sc):
        thinking5, answer5 = await cot_sc_agents_5[i]([taskInfo, thinking4, answer4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_5[i].id}, matching coupling pattern to choices, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    most_common_answer_5 = Counter(possible_answers_5).most_common(1)[0][0]
    thinking5 = thinkingmapping_5[most_common_answer_5]
    answer5 = answermapping_5[most_common_answer_5]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs