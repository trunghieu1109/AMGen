async def forward_157(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Understand the biological context and key molecular events described in the query: the inactive transcription factor subunit, its activation by phosphorylation on Ser residues in the transactivation domain, subsequent dimerization, nuclear translocation, and role in gene transcription. Ensure comprehension of the functional domains involved (transactivation and dimerization domains)."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding biological context, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Analyze the nature and molecular consequences of the two mutations described: mutation X (a recessive loss-of-function missense mutation in the transactivation domain) and mutation Y (a heterozygous dominant-negative mutation in the dimerization domain). Include brief consideration of how these mutations affect protein function and interaction. Briefly mention each answer choice (A-D) to keep all options in consideration."
    N2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing mutations, thinking: {thinking2.content}; answer: {answer2.content}")
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
    cot_reflect_instruction_3 = "Sub-task 3: Explain the molecular mechanisms of dominant-negative mutations, focusing on how a mutation in the dimerization domain (mutation Y) can interfere with the function of the wild-type protein. Explicitly reference the possible molecular phenotypes listed in the answer choices (loss of dimerization, aggregation, degradation, conformational change) to keep all options in consideration."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, explaining dominant-negative mechanism, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the explanation of dominant-negative mutation mechanism and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining explanation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    debate_instruction_4a = "Sub-task 4a: Perform a structured debate-style analysis of each answer choice (A–D). For each choice, generate pros and cons based on the biological context and dominant-negative mutation mechanisms, explicitly comparing key mechanisms such as protein aggregation versus degradation and their expected phenotypic outcomes."
    debate_agents_4a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4a = self.max_round
    all_thinking4a = [[] for _ in range(N_max_4a)]
    all_answer4a = [[] for _ in range(N_max_4a)]
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": debate_instruction_4a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "answer choices: A-loss of dimerization and wild-type phenotype, B-protein aggregation and loss-of-function phenotype, C-protein degradation and loss-of-function of the wild-type allele, D-change of protein conformation and gain-of-function phenotype"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4a):
        for i, agent in enumerate(debate_agents_4a):
            if r == 0:
                thinking4a, answer4a = await agent([taskInfo, thinking3, answer3], debate_instruction_4a, r, is_sub_task=True)
            else:
                input_infos_4a = [taskInfo, thinking3, answer3] + all_thinking4a[r-1] + all_answer4a[r-1]
                thinking4a, answer4a = await agent(input_infos_4a, debate_instruction_4a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating pros and cons of answer choices, thinking: {thinking4a.content}; answer: {answer4a.content}")
            all_thinking4a[r].append(thinking4a)
            all_answer4a[r].append(answer4a)
    sub_tasks.append(f"Sub-task 4a output: thinking - {all_thinking4a[-1]}; answer - {all_answer4a[-1]}")
    subtask_desc4a['response'] = {
        "thinking": all_thinking4a[-1],
        "answer": all_answer4a[-1]
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    debate_instruction_4b = "Sub-task 4b: Select the most likely molecular phenotype observed in the presence of mutation Y by directly comparing the debated answer choices. Explicitly connect the reasoning to the biological context, mutation effects, and dominant-negative mechanisms, ensuring a clear justification for the final choice. Return only the letter (A, B, C, or D) corresponding to the correct answer choice."
    final_decision_agent_4b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4b, answer4b = await final_decision_agent_4b([taskInfo] + all_thinking4a[-1] + all_answer4a[-1], debate_instruction_4b, is_sub_task=True)
    agents.append(f"Final Decision agent, selecting molecular phenotype, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": debate_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "Final Decision"
    }
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    reflexion_instruction_5 = "Sub-task 5: Review and reflect on the reasoning process and final selection to ensure completeness and correctness. Confirm that the output format strictly adheres to the requirement of returning only the letter (A, B, C, or D) corresponding to the correct answer choice for the multiple-choice question."
    reflexion_agent_5 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    reflexion_inputs_5 = [taskInfo, thinking4b, answer4b]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": reflexion_instruction_5,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await reflexion_agent_5(reflexion_inputs_5, reflexion_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion agent, reviewing final selection, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "please verify that the final answer is correct and strictly formatted as a single letter (A, B, C, or D).", i, is_sub_task=True)
        agents.append(f"Critic agent, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        reflexion_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await reflexion_agent_5(reflexion_inputs_5, reflexion_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion agent, refining final selection, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs