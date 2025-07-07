async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Analyze the first Michael addition reaction involving dimethyl malonate, methyl (E)-3-(p-tolyl)acrylate, and NaOEt/EtOH to identify the nucleophile and electrophile, determine the site of nucleophilic attack (β-position), and predict the resonance-stabilized intermediate formed after the attack."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing first Michael addition reaction, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)

    cot_sc_instruction_2 = "Sub-task 2: Predict the major final product (A) of the first Michael addition reaction by considering the intermediate from subtask_1, including enolate formation, protonation or tautomerization steps, and explicitly assign regiochemistry and stereochemistry. Generate at least three plausible product structures and select the most consistent one using self-consistency chain-of-thought reasoning."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, predicting major final product (A), thinking: {thinking2.content}; answer: {answer2.content}")
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

    cot_instruction_3 = "Sub-task 3: Cross-check the predicted product (A) against the multiple-choice options for product A to verify consistency in structure and nomenclature, identifying any mismatches early."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "multiple-choice options for product A"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, cross-checking predicted product (A) with choices, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)

    cot_instruction_4 = "Sub-task 4: Analyze the second Michael addition reaction involving 1-(cyclohex-1-en-1-yl)piperidine, (E)-but-2-enenitrile, and MeOH/H3O+ to identify the nucleophile and electrophile, determine the site of nucleophilic attack (β-position), and predict the resonance-stabilized intermediate formed after the attack."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, analyzing second Michael addition reaction, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)

    cot_sc_instruction_5 = "Sub-task 5: Predict the major final product (B) of the second Michael addition reaction by considering the intermediate from subtask_4, including enolate formation, protonation, and possible tautomerization under acidic conditions. Generate at least three plausible product structures and select the most consistent one using self-consistency chain-of-thought reasoning."
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking4, answer4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, predicting major final product (B), thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    answer5_content = Counter(possible_answers_5).most_common(1)[0][0]
    thinking5 = thinkingmapping_5[answer5_content]
    answer5 = answermapping_5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)

    cot_instruction_6 = "Sub-task 6: Cross-check the predicted product (B) against the multiple-choice options for product B to verify consistency in structure and nomenclature, identifying any mismatches early."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5", "multiple-choice options for product B"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, cross-checking predicted product (B) with choices, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)

    debate_instruction_7 = "Sub-task 7: Identify the unknown reactant C in the third Michael addition reaction with but-3-en-2-one and KOH/H2O, based on the given final product 2-(3-oxobutyl)cyclohexane-1,3-dione and typical Michael addition mechanisms. Generate at least three plausible candidates for reactant C and select the most consistent one using self-consistency chain-of-thought reasoning."
    N = self.max_sc
    cot_agents_7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_7 = []
    thinkingmapping_7 = {}
    answermapping_7 = {}
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_7,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking7, answer7 = await cot_agents_7[i]([taskInfo], debate_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_7[i].id}, identifying unknown reactant C, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers_7.append(answer7.content)
        thinkingmapping_7[answer7.content] = thinking7
        answermapping_7[answer7.content] = answer7
    answer7_content = Counter(possible_answers_7).most_common(1)[0][0]
    thinking7 = thinkingmapping_7[answer7_content]
    answer7 = answermapping_7[answer7_content]
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)

    cot_instruction_8 = "Sub-task 8: Analyze the third Michael addition reaction mechanism to confirm the formation of the final product from reactant C and but-3-en-2-one under basic aqueous conditions, including enolate formation and conjugate addition, and predict the final product (C) structure with explicit consideration of tautomerization or protonation states."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, analyzing third Michael addition mechanism, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {
        "thinking": thinking8,
        "answer": answer8
    }
    logs.append(subtask_desc8)

    cot_instruction_9 = "Sub-task 9: Cross-check the predicted product (C) against the multiple-choice options for product C to verify consistency in structure and nomenclature, identifying any mismatches early."
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_9,
        "context": ["user query", "thinking of subtask 8", "answer of subtask 8", "multiple-choice options for product C"],
        "agent_collaboration": "CoT"
    }
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking8, answer8], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, cross-checking predicted product (C) with choices, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {
        "thinking": thinking9,
        "answer": answer9
    }
    logs.append(subtask_desc9)

    debate_instruction_10 = "Sub-task 10: Systematically compare the predicted products (A, B, and C) from subtasks 3, 6, and 9 with all given multiple-choice options. Use a structured checklist or table format to verify the correctness of each product assignment per choice, ensuring full alignment before final selection."
    debate_agents_10 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_10 = self.max_round
    all_thinking10 = [[] for _ in range(N_max_10)]
    all_answer10 = [[] for _ in range(N_max_10)]
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": debate_instruction_10,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 6", "answer of subtask 6", "thinking of subtask 9", "answer of subtask 9", "multiple-choice options"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_10):
        for i, agent in enumerate(debate_agents_10):
            if r == 0:
                thinking10, answer10 = await agent([taskInfo, thinking3, answer3, thinking6, answer6, thinking9, answer9], debate_instruction_10, r, is_sub_task=True)
            else:
                input_infos_10 = [taskInfo, thinking3, answer3, thinking6, answer6, thinking9, answer9] + all_thinking10[r-1] + all_answer10[r-1]
                thinking10, answer10 = await agent(input_infos_10, debate_instruction_10, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing predicted products with choices, thinking: {thinking10.content}; answer: {answer10.content}")
            all_thinking10[r].append(thinking10)
            all_answer10[r].append(answer10)
    final_decision_agent_10 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking10, answer10 = await final_decision_agent_10([taskInfo] + all_thinking10[-1] + all_answer10[-1], "Sub-task 10: Make final decision on the correct matching choice for products A, B, and C.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct choice, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {
        "thinking": thinking10,
        "answer": answer10
    }
    logs.append(subtask_desc10)

    cot_reflect_instruction_11 = "Sub-task 11: Perform a reflexive self-audit of the final selected answer by reviewing each predicted product against the chosen multiple-choice option to confirm complete agreement. If discrepancies are found, re-evaluate the relevant subtasks and update predictions accordingly."
    cot_agent_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_11 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_11 = self.max_round
    cot_inputs_11 = [taskInfo, thinking3, answer3, thinking6, answer6, thinking9, answer9, thinking10, answer10]
    subtask_desc11 = {
        "subtask_id": "subtask_11",
        "instruction": cot_reflect_instruction_11,
        "context": ["user query", "thinking and answer of subtasks 3, 6, 9, 10"],
        "agent_collaboration": "Reflexion"
    }
    thinking11, answer11 = await cot_agent_11(cot_inputs_11, cot_reflect_instruction_11, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_11.id}, performing self-audit of final answer, thinking: {thinking11.content}; answer: {answer11.content}")
    for i in range(N_max_11):
        feedback, correct = await critic_agent_11([taskInfo, thinking11, answer11], "please review the self-audit of the final selected answer and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_11.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_11.extend([thinking11, answer11, feedback])
        thinking11, answer11 = await cot_agent_11(cot_inputs_11, cot_reflect_instruction_11, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_11.id}, refining self-audit of final answer, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    subtask_desc11['response'] = {
        "thinking": thinking11,
        "answer": answer11
    }
    logs.append(subtask_desc11)

    final_answer = await self.make_final_answer(thinking11, answer11, sub_tasks, agents)
    return final_answer, logs
