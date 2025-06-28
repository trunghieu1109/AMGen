async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_agents_sc = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    # Stage 1: Analyze reaction components and conditions
    cot_instruction_1 = "Sub-task 1: Analyze reaction (A) components: dimethyl malonate, methyl (E)-3-(p-tolyl)acrylate, and (NaOEt, EtOH). Identify nucleophile, electrophile, reaction conditions, and determine the site of nucleophilic attack and intermediate formation, including resonance stabilization and expected tautomeric forms."
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_agents_sc[i]([taskInfo], cot_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, analyzing reaction (A), thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1.content)
        thinkingmapping_1[answer1.content] = thinking1
        answermapping_1[answer1.content] = answer1
    answer1_content = Counter(possible_answers_1).most_common(1)[0][0]
    thinking1 = thinkingmapping_1[answer1_content]
    answer1 = answermapping_1[answer1_content]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_instruction_2 = "Sub-task 2: Analyze reaction (B) components: 1-(cyclohex-1-en-1-yl)piperidine, (E)-but-2-enenitrile, and (MeOH, H3O+). Identify nucleophile, electrophile, and reaction conditions. Detail the Michael addition mechanism including formation of enamine intermediate, acid-promoted hydrolysis, and predict the stable tautomeric form of the product."
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_agents_sc[i]([taskInfo], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, analyzing reaction (B), thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction_3 = "Sub-task 3: Analyze reaction (C) components: unknown reactant C, but-3-en-2-one, and (KOH, H2O). Deduce the structure of C based on the product name 2-(3-oxobutyl)cyclohexane-1,3-dione. Identify nucleophile and electrophile roles, and predict the intermediate formed during the Michael addition."
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(self.max_sc):
        thinking3, answer3 = await cot_agents_sc[i]([taskInfo], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, analyzing reaction (C), thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Stage 2: Predict products
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4 = [taskInfo, thinking1, answer1]
    cot_instruction_4 = "Sub-task 4: Predict the major final product of reaction (A) by applying the Michael addition mechanism: nucleophilic attack of dimethyl malonate on the β-carbon of methyl (E)-3-(p-tolyl)acrylate under basic conditions. Assign precise carbon numbering and substituent positions, consider resonance stabilization, and identify the most stable tautomeric form of the product."
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, predicting product for reaction (A), thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "please review the predicted product for reaction (A) and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining product prediction for reaction (A), thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": "Sub-task 5a: Detail the mechanistic pathway of reaction (B), focusing on the enamine intermediate formation from 1-(cyclohex-1-en-1-yl)piperidine and (E)-but-2-enenitrile, followed by acid-promoted hydrolysis under (MeOH, H3O+) conditions. Identify all possible intermediates and their transformations.",
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_5a = []
    thinkingmapping_5a = {}
    answermapping_5a = {}
    for i in range(self.max_sc):
        thinking5a, answer5a = await cot_agents_sc[i]([taskInfo, thinking2, answer2], subtask_desc5a["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, detailing mechanism for reaction (B), thinking: {thinking5a.content}; answer: {answer5a.content}")
        possible_answers_5a.append(answer5a.content)
        thinkingmapping_5a[answer5a.content] = thinking5a
        answermapping_5a[answer5a.content] = answer5a
    answer5a_content = Counter(possible_answers_5a).most_common(1)[0][0]
    thinking5a = thinkingmapping_5a[answer5a_content]
    answer5a = answermapping_5a[answer5a_content]
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a["response"] = {"thinking": thinking5a, "answer": answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_5b = [taskInfo, thinking2, answer2, thinking5a, answer5a]
    cot_instruction_5b = "Sub-task 5b: Predict the major final product of reaction (B) by determining the stable tautomeric form after hydrolysis and protonation steps. Assign correct carbon numbering and substituent positions, and verify the product structure aligns with chemical logic and reaction conditions."
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_instruction_5b,
        "context": ["user query", thinking2.content, answer2.content, thinking5a.content, answer5a.content],
        "agent_collaboration": "Reflexion"
    }
    thinking5b, answer5b = await cot_agent_5b(cot_inputs_5b, cot_instruction_5b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5b.id}, predicting product for reaction (B), thinking: {thinking5b.content}; answer: {answer5b.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_5b([taskInfo, thinking5b, answer5b], "please review the predicted product for reaction (B) and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5b.extend([thinking5b, answer5b, feedback])
        thinking5b, answer5b = await cot_agent_5b(cot_inputs_5b, cot_instruction_5b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5b.id}, refining product prediction for reaction (B), thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b["response"] = {"thinking": thinking5b, "answer": answer5b}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    cot_agent_6a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": "Sub-task 6a: Predict the intermediate formed in reaction (C) by applying the Michael addition mechanism: nucleophilic attack of C (likely cyclohexane-1,3-dione or its enolate) on but-3-en-2-one under basic aqueous conditions (KOH, H2O).",
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_6a = []
    thinkingmapping_6a = {}
    answermapping_6a = {}
    for i in range(self.max_sc):
        thinking6a, answer6a = await cot_agents_sc[i]([taskInfo, thinking3, answer3], subtask_desc6a["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, predicting intermediate for reaction (C), thinking: {thinking6a.content}; answer: {answer6a.content}")
        possible_answers_6a.append(answer6a.content)
        thinkingmapping_6a[answer6a.content] = thinking6a
        answermapping_6a[answer6a.content] = answer6a
    answer6a_content = Counter(possible_answers_6a).most_common(1)[0][0]
    thinking6a = thinkingmapping_6a[answer6a_content]
    answer6a = answermapping_6a[answer6a_content]
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a["response"] = {"thinking": thinking6a, "answer": answer6a}
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])
    cot_agent_6b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_6b = [taskInfo, thinking3, answer3, thinking6a, answer6a]
    cot_instruction_6b = "Sub-task 6b: Determine the major final product of reaction (C), considering tautomerism and enol-keto equilibria, and confirm the product structure as 2-(3-oxobutyl)cyclohexane-1,3-dione or its thermodynamically favored tautomeric form."
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": cot_instruction_6b,
        "context": ["user query", thinking3.content, answer3.content, thinking6a.content, answer6a.content],
        "agent_collaboration": "Reflexion"
    }
    thinking6b, answer6b = await cot_agent_6b(cot_inputs_6b, cot_instruction_6b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6b.id}, predicting product for reaction (C), thinking: {thinking6b.content}; answer: {answer6b.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_6b([taskInfo, thinking6b, answer6b], "please review the predicted product for reaction (C) and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6b.extend([thinking6b, answer6b, feedback])
        thinking6b, answer6b = await cot_agent_6b(cot_inputs_6b, cot_instruction_6b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6b.id}, refining product prediction for reaction (C), thinking: {thinking6b.content}; answer: {answer6b.content}")
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b["response"] = {"thinking": thinking6b, "answer": answer6b}
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])
    # Stage 3: Analyze and match predicted products to choices
    cot_instruction_7a = "Sub-task 7a: Analyze the predicted products from subtasks 4, 5b, and 6b with respect to tautomerism, enol-keto equilibria, and naming conventions. Prepare detailed structural and nomenclature comparisons to the multiple-choice options, highlighting possible naming variants and tautomeric forms."
    subtask_desc7a = {
        "subtask_id": "subtask_7a",
        "instruction": cot_instruction_7a,
        "context": ["user query", thinking4.content, answer4.content, thinking5b.content, answer5b.content, thinking6b.content, answer6b.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_7a = []
    thinkingmapping_7a = {}
    answermapping_7a = {}
    for i in range(self.max_sc):
        thinking7a, answer7a = await cot_agents_sc[i]([taskInfo, thinking4, answer4, thinking5b, answer5b, thinking6b, answer6b], cot_instruction_7a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, analyzing product naming and tautomerism, thinking: {thinking7a.content}; answer: {answer7a.content}")
        possible_answers_7a.append(answer7a.content)
        thinkingmapping_7a[answer7a.content] = thinking7a
        answermapping_7a[answer7a.content] = answer7a
    answer7a_content = Counter(possible_answers_7a).most_common(1)[0][0]
    thinking7a = thinkingmapping_7a[answer7a_content]
    answer7a = answermapping_7a[answer7a_content]
    sub_tasks.append(f"Sub-task 7a output: thinking - {thinking7a.content}; answer - {answer7a.content}")
    subtask_desc7a["response"] = {"thinking": thinking7a, "answer": answer7a}
    logs.append(subtask_desc7a)
    print("Step 7a: ", sub_tasks[-1])
    debate_instruction_7b = "Sub-task 7b: Match the predicted products to the given multiple-choice options (choices 1 to 4) by debating and adjudicating the best fit considering all chemical reasoning, tautomeric forms, and naming conventions. Provide a reasoned justification for the selected choice."
    subtask_desc7b = {
        "subtask_id": "subtask_7b",
        "instruction": debate_instruction_7b,
        "context": ["user query", thinking4.content, answer4.content, thinking5b.content, answer5b.content, thinking6b.content, answer6b.content, thinking7a.content, answer7a.content],
        "agent_collaboration": "Debate"
    }
    all_thinking7b = [[] for _ in range(N_max)]
    all_answer7b = [[] for _ in range(N_max)]
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking7b, answer7b = await agent([taskInfo, thinking4, answer4, thinking5b, answer5b, thinking6b, answer6b, thinking7a, answer7a], debate_instruction_7b, r, is_sub_task=True)
            else:
                input_infos_7b = [taskInfo, thinking4, answer4, thinking5b, answer5b, thinking6b, answer6b, thinking7a, answer7a] + all_thinking7b[r-1] + all_answer7b[r-1]
                thinking7b, answer7b = await agent(input_infos_7b, debate_instruction_7b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating product-choice matching, thinking: {thinking7b.content}; answer: {answer7b.content}")
            all_thinking7b[r].append(thinking7b)
            all_answer7b[r].append(answer7b)
    final_decision_agent_7b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7b, answer7b = await final_decision_agent_7b([taskInfo] + all_thinking7b[-1] + all_answer7b[-1], "Sub-task 7b: Make final decision on the correct multiple-choice answer for reactions A, B, and C.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final choice, thinking: {thinking7b.content}; answer: {answer7b.content}")
    sub_tasks.append(f"Sub-task 7b output: thinking - {thinking7b.content}; answer - {answer7b.content}")
    subtask_desc7b["response"] = {"thinking": thinking7b, "answer": answer7b}
    logs.append(subtask_desc7b)
    print("Step 7b: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7b, answer7b, sub_tasks, agents)
    return final_answer, logs