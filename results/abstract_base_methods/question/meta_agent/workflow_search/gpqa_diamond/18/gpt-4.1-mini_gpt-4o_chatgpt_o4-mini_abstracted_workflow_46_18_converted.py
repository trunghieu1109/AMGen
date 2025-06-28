async def forward_18(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_instruction_1 = "Sub-task 1: Analyze the Michael reaction mechanism in detail, focusing on nucleophilic addition to α,β-unsaturated carbonyl compounds. Identify roles of reagents (NaOEt, NaH, THF), nucleophile and electrophile nature, mechanistic steps including enolate formation, regioselectivity, and stereochemical considerations. Include pKa and resonance effects relevant to substrates."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_sc_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, detailed Michael reaction mechanism analysis, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Identify and characterize starting materials and reagents for reactions A and B. List all functional groups, acidic α-hydrogens, and possible enolate formation sites. Rank α-hydrogen acidities with justification based on pKa values and resonance stabilization to determine the most plausible enolate species. Use outputs from Sub-task 1."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identifying and ranking α-hydrogens, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_instruction_3a = "Sub-task 3a: For reaction A, determine the most likely enolate formed from methyl 2-oxocyclohexane-1-carboxylate under NaOEt conditions. Analyze α-hydrogen acidity, resonance stabilization, and sketch plausible enolate structures with nucleophilic sites. Use outputs from Sub-task 2."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2], cot_sc_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, determining enolate for reaction A, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a["response"] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    cot_sc_instruction_3b = "Sub-task 4a: For reaction B, determine the most likely enolate formed from ethyl 2-ethylbutanoate under NaH conditions. Analyze α-hydrogen acidity, resonance stabilization, and sketch plausible enolate structures with nucleophilic sites. Use outputs from Sub-task 2."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking2, answer2], cot_sc_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, determining enolate for reaction B, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b["response"] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 4a: ", sub_tasks[-1])
    reflexion_instruction_3c = "Sub-task 3c: Reflect on and verify the enolate structures and nucleophilic sites proposed for reactions A and B. Check for consistency with acidity rankings, resonance stabilization, and mechanistic plausibility. Revise if inconsistencies are found."
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3c = self.max_round
    cot_inputs_3c = [taskInfo, thinking3a, answer3a, thinking3b, answer3b]
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": reflexion_instruction_3c,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Reflexion"
    }
    thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, reflexion_instruction_3c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, verifying enolate structures, thinking: {thinking3c.content}; answer: {answer3c.content}")
    for i in range(N_max_3c):
        feedback, correct = await critic_agent_3c([taskInfo, thinking3c, answer3c], "Review the enolate structures and nucleophilic sites for consistency and mechanistic plausibility.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3c.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_3c.extend([thinking3c, answer3c, feedback])
        thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, reflexion_instruction_3c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, refining enolate verification, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c["response"] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])
    cot_sc_instruction_4 = "Sub-task 5: Predict the regio- and stereochemistry of the Michael addition products for reactions A and B by combining the verified enolate nucleophiles and electrophilic Michael acceptors. Include detailed mechanistic reasoning, intermediate structures, and stereochemical outcomes. Use outputs from Sub-task 3c."
    N4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N4):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3c, answer3c], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, predicting Michael addition products, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    most_common_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[most_common_answer_4]
    answer4 = answermapping_4[most_common_answer_4]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 5: ", sub_tasks[-1])
    reflexion_instruction_5 = "Sub-task 6: Cross-verify the predicted products for reactions A and B by checking mass balance, functional group counts, and stereochemical feasibility. Reflect on alternative mechanistic pathways or enolate formations and revise predictions if inconsistencies are found. Use outputs from Sub-task 5."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_6",
        "instruction": reflexion_instruction_5,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflexion_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, verifying predicted products, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "Review predicted products for mass balance, functional groups, stereochemistry, and alternative pathways.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflexion_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining product verification, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 6: ", sub_tasks[-1])
    debate_instruction_6 = "Sub-task 7: Compare the verified predicted products of reactions A and B against the provided multiple-choice options. Analyze substituents, positions, stereochemistry, and functional groups to identify the correct match. Use outputs from Sub-task 6."
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing predicted products with choices, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 7: Make final decision on the correct multiple-choice match for products A and B.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct product match, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 7: ", sub_tasks[-1])
    debate_instruction_7 = "Sub-task 8: Select and return the letter (A, B, C, or D) corresponding to the choice that correctly identifies the final products of both reactions A and B based on the detailed comparison and verification from Sub-task 7."
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    subtask_desc7 = {
        "subtask_id": "subtask_8",
        "instruction": debate_instruction_7,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct choice letter, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 8: Make final decision on the correct letter choice for the products.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing correct letter choice, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 8: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs