async def forward_94(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Analyze the molecular structure and functional groups of 3,3,6-trimethylhepta-1,5-dien-4-one to identify all reactive sites relevant for epoxidation and subsequent nucleophilic attack, explicitly noting the positions and nature of the two alkenes and the ketone."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing molecular structure and reactive sites, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Determine the products formed when 3,3,6-trimethylhepta-1,5-dien-4-one is treated with exactly 1 equivalent of meta-chloroperbenzoic acid (mCPBA), explicitly considering that only one alkene is epoxidized while the other alkene remains intact. Generate all plausible epoxide isomers, name them accurately reflecting the retention of the second alkene (e.g., using correct unsaturation suffixes), and rationalize the formation of two different products in approximately 1:1 ratio using a Self-Consistency Chain-of-Thought (SC CoT) approach to validate the degree of unsaturation and structural consistency, based on Sub-task 1 output."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining epoxide products with retention of one alkene, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    validation_instruction_2 = "Sub-task 2 Validation: Validate the predicted epoxide products from Sub-task 2 by comparing their degrees of unsaturation and structural features against the starting material minus one alkene. Confirm that one alkene remains intact and that product names reflect this. If inconsistencies are found, request re-evaluation."
    cot_agent_val_2 = LLMAgentBase(["thinking", "answer"], "Validation Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_val = {
        "subtask_id": "subtask_2_validation",
        "instruction": validation_instruction_2,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Validation"
    }
    thinking_val2, answer_val2 = await cot_agent_val_2([taskInfo, thinking2, answer2], validation_instruction_2, is_sub_task=True)
    agents.append(f"Validation agent {cot_agent_val_2.id}, validating epoxide products, thinking: {thinking_val2.content}; answer: {answer_val2.content}")
    sub_tasks.append(f"Sub-task 2 Validation output: thinking - {thinking_val2.content}; answer - {answer_val2.content}")
    subtask_desc2_val['response'] = {"thinking": thinking_val2, "answer": answer_val2}
    logs.append(subtask_desc2_val)
    print("Step 2 Validation: ", sub_tasks[-1])
    if "inconsistency" in answer_val2.content.lower() or "error" in answer_val2.content.lower():
        for i in range(N):
            thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
            agents.append(f"CoT-SC agent {cot_agents_2[i].id}, re-determining epoxide products after validation failure, thinking: {thinking2.content}; answer: {answer2.content}")
            possible_answers_2.append(answer2.content)
            thinkingmapping_2[answer2.content] = thinking2
            answermapping_2[answer2.content] = answer2
        most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
        thinking2 = thinkingmapping_2[most_common_answer_2]
        answer2 = answermapping_2[most_common_answer_2]
        sub_tasks.append(f"Sub-task 2 output (revised): thinking - {thinking2.content}; answer - {answer2.content}")
        subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
        logs.append(subtask_desc2)
        print("Step 2 (revised): ", sub_tasks[-1])
    cot_instruction_3 = "Sub-task 3: Analyze the reaction of methyllithium with copper(I) iodide to form the Gilman reagent (organocuprate), detailing its formation, reactivity, and selectivity towards epoxides and other electrophilic sites such as ketones."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing Gilman reagent formation and reactivity, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction_4a = "Sub-task 4a: Predict the regioselectivity of the nucleophilic ring-opening of the epoxide(s) formed in Sub-task 2 by the Gilman reagent, considering the presence of the remaining alkene and ketone functional groups."
    cot_instruction_4b = "Sub-task 4b: Predict the stereochemical outcomes of the nucleophilic ring-opening reactions, explicitly considering stereochemistry at the epoxide carbons and any new stereocenters formed, including possible stereochemical retention or inversion."
    cot_instruction_4c = "Sub-task 4c: Evaluate potential side reactions or competing pathways involving the ketone or the remaining alkene under the reaction conditions with the Gilman reagent, and assess their impact on the product distribution."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_4c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking of subtask 2 validation", "answer of subtask 2 validation", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4a, answer4a = await cot_agent_4a([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4a.id}, predicting regioselectivity of nucleophilic ring-opening, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "CoT"
    }
    thinking4b, answer4b = await cot_agent_4b([taskInfo, thinking4a, answer4a], cot_instruction_4b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4b.id}, predicting stereochemical outcomes of nucleophilic ring-opening, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_instruction_4c,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "CoT"
    }
    thinking4c, answer4c = await cot_agent_4c([taskInfo, thinking4b, answer4b], cot_instruction_4c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4c.id}, evaluating side reactions and competing pathways, thinking: {thinking4c.content}; answer: {answer4c.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {"thinking": thinking4c, "answer": answer4c}
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])
    cot_reflect_instruction_4 = "Sub-task 4 Reflexion: Review and refine the predictions from Sub-tasks 4a, 4b, and 4c, ensuring chemical accuracy, stereochemical detail, and consideration of side reactions before finalizing the predicted product structures."
    cot_agent_4r = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4r = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4r = self.max_round
    cot_inputs_4r = [taskInfo, thinking4a, answer4a, thinking4b, answer4b, thinking4c, answer4c]
    subtask_desc4r = {
        "subtask_id": "subtask_4_reflexion",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking and answer of subtask 4a", "thinking and answer of subtask 4b", "thinking and answer of subtask 4c"],
        "agent_collaboration": "Reflexion"
    }
    thinking4r, answer4r = await cot_agent_4r(cot_inputs_4r, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4r.id}, refining predictions of nucleophilic ring-opening and side reactions, thinking: {thinking4r.content}; answer: {answer4r.content}")
    for i in range(N_max_4r):
        feedback, correct = await critic_agent_4r([taskInfo, thinking4r, answer4r], "please review the predicted product structures and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4r.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4r.extend([thinking4r, answer4r, feedback])
        thinking4r, answer4r = await cot_agent_4r(cot_inputs_4r, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4r.id}, refining predictions, thinking: {thinking4r.content}; answer: {answer4r.content}")
    sub_tasks.append(f"Sub-task 4 Reflexion output: thinking - {thinking4r.content}; answer - {answer4r.content}")
    subtask_desc4r['response'] = {"thinking": thinking4r, "answer": answer4r}
    logs.append(subtask_desc4r)
    print("Step 4 Reflexion: ", sub_tasks[-1])
    debate_instruction_5 = "Sub-task 5: Critically compare the predicted product structures from Sub-task 4 Reflexion with the given multiple-choice options, evaluating each choice for chemical consistency, stereochemistry, and structural features, and provide detailed justification for acceptance or rejection of each option."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4 reflexion", "answer of subtask 4 reflexion"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4r, answer4r], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4r, answer4r] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing products, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 6: Select the correct product name from the multiple-choice options or conclude none match based on Sub-task 5 analysis.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct product, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": "Select the correct product name from the multiple-choice options or conclude none match.",
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Final Decision"
    }
    subtask_desc6['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs