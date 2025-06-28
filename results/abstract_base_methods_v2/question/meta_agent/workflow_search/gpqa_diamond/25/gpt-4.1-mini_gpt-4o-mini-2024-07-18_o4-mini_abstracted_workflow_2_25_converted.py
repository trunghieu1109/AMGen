async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_sc_instruction_0 = (
        "Sub-task 0: Identify and classify the reaction mechanism involved in the transformation of cyclohexene and reactant (A) to 8,8-diiodobicyclo[4.2.0]octan-7-one by explicitly analyzing bond-making and bond-breaking events, "
        "considering ketene [2+2] cycloaddition versus electrophilic addition mechanisms. Use a self-consistency chain-of-thought (SC CoT) approach where multiple agents propose, justify, and select the most plausible mechanism."
    )
    N0 = self.max_sc
    cot_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N0)]
    possible_answers_0 = []
    thinkingmapping_0 = {}
    answermapping_0 = {}
    subtask_desc0 = {
        "subtask_id": "subtask_0",
        "instruction": cot_sc_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N0):
        thinking0, answer0 = await cot_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0[i].id}, identifying reaction mechanism, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_0.append(answer0.content)
        thinkingmapping_0[answer0.content] = thinking0
        answermapping_0[answer0.content] = answer0
    answer0_content = Counter(possible_answers_0).most_common(1)[0][0]
    thinking0 = thinkingmapping_0[answer0_content]
    answer0 = answermapping_0[answer0_content]
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {
        "thinking": thinking0,
        "answer": answer0
    }
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])
    
    cot_instruction_1a = (
        "Sub-task 1a: Analyze the structural features of the possible reactants (A) given in the options, focusing on their compatibility with the reaction mechanism identified in Subtask 0. "
        "Consider functional groups, ring strain, and electronic factors."
    )
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query", "thinking of subtask_0", "answer of subtask_0"],
        "agent_collaboration": "CoT"
    }
    thinking1a, answer1a = await cot_agent_1a([taskInfo, thinking0, answer0], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, analyzing reactants (A) structural features, thinking: {thinking1a.content}; answer: {answer1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc1a['response'] = {
        "thinking": thinking1a,
        "answer": answer1a
    }
    logs.append(subtask_desc1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_instruction_1b = (
        "Sub-task 1b: Validate the compatibility of each possible reactant (A) with the reaction mechanism identified in Subtask 0. "
        "Confirm which reactant (A) can feasibly react with cyclohexene to form the specified product, considering mechanistic details and electronic factors."
    )
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask_0", "answer of subtask_0", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "CoT"
    }
    thinking1b, answer1b = await cot_agent_1b([taskInfo, thinking0, answer0, thinking1a, answer1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, validating reactants (A) compatibility, thinking: {thinking1b.content}; answer: {answer1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc1b['response'] = {
        "thinking": thinking1b,
        "answer": answer1b
    }
    logs.append(subtask_desc1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_instruction_2 = (
        "Sub-task 2: Analyze the four given dienes (B) focusing on their structural features, conjugation, substitution patterns, stereochemistry (including s-cis/s-trans conformations), "
        "and pericyclic reaction principles to understand factors influencing their reactivity."
    )
    N2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing dienes (B), thinking: {thinking2.content}; answer: {answer2.content}")
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
    
    cot_sc_instruction_3 = (
        "Sub-task 3: Generate multiple independent reactivity rankings of the given dienes (B) using self-consistency chain-of-thought (SC CoT) reasoning, "
        "explicitly incorporating conformational analysis and known organic chemistry principles, then perform majority voting or consistency checks to select the most chemically sound reactivity sequence from most reactive to least reactive."
    )
    N3 = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask_2", "answer of subtask_2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N3):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, generating diene reactivity rankings, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_reflect_instruction_4 = (
        "Sub-task 4: Cross-validate and reconcile the diene reactivity ranking obtained in Subtask 3 with the structural and mechanistic insights from Subtasks 0 and 1, "
        "identifying and resolving any discrepancies or inconsistencies to ensure a coherent and chemically accurate reactivity sequence."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking0, answer0, thinking1b, answer1b, thinking3, answer3]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking of subtask_0", "answer of subtask_0", "thinking of subtask_1b", "answer of subtask_1b", "thinking of subtask_3", "answer of subtask_3"],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, cross-validating diene reactivity ranking, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "Please review the diene reactivity ranking cross-validation and provide limitations or corrections.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining diene reactivity ranking, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5a = (
        "Sub-task 5a: Cross-validate the candidate reactant (A) identified in Subtask 1b and the diene reactivity sequence from Subtask 4 against all multiple-choice options, "
        "checking for exact consistency between the mechanistic feasibility, reactivity ranking, and the options provided."
    )
    debate_agents_5a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5a = self.max_round
    all_thinking5a = [[] for _ in range(N_max_5a)]
    all_answer5a = [[] for _ in range(N_max_5a)]
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": debate_instruction_5a,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5a):
        for i, agent in enumerate(debate_agents_5a):
            if r == 0:
                thinking5a, answer5a = await agent([taskInfo, thinking1b, answer1b, thinking4, answer4], debate_instruction_5a, r, is_sub_task=True)
            else:
                input_infos_5a = [taskInfo, thinking1b, answer1b, thinking4, answer4] + all_thinking5a[r-1] + all_answer5a[r-1]
                thinking5a, answer5a = await agent(input_infos_5a, debate_instruction_5a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, cross-validating reactant and diene ranking, thinking: {thinking5a.content}; answer: {answer5a.content}")
            all_thinking5a[r].append(thinking5a)
            all_answer5a[r].append(answer5a)
    sub_tasks.append(f"Sub-task 5a output: thinking - {all_thinking5a[-1]}; answer - {all_answer5a[-1]}")
    subtask_desc5a['response'] = {
        "thinking": all_thinking5a[-1],
        "answer": all_answer5a[-1]
    }
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    debate_instruction_5b = (
        "Sub-task 5b: Select the final correct multiple-choice answer (A, B, C, or D) only after confirming full consistency between the reasoning outputs (reactant identification and diene reactivity sequence) "
        "and the given options, ensuring no contradictions remain. Provide a reasoned final decision."
    )
    debate_agents_5b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5b = self.max_round
    all_thinking5b = [[] for _ in range(N_max_5b)]
    all_answer5b = [[] for _ in range(N_max_5b)]
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": debate_instruction_5b,
        "context": ["user query", "thinking of subtask_5a", "answer of subtask_5a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5b):
        for i, agent in enumerate(debate_agents_5b):
            if r == 0:
                thinking5b, answer5b = await agent([taskInfo] + all_thinking5a[-1] + all_answer5a[-1], debate_instruction_5b, r, is_sub_task=True)
            else:
                input_infos_5b = [taskInfo] + all_thinking5a[-1] + all_answer5a[-1] + all_thinking5b[r-1] + all_answer5b[r-1]
                thinking5b, answer5b = await agent(input_infos_5b, debate_instruction_5b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting final answer, thinking: {thinking5b.content}; answer: {answer5b.content}")
            all_thinking5b[r].append(thinking5b)
            all_answer5b[r].append(answer5b)
    final_decision_agent_5b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5b, answer5b = await final_decision_agent_5b([taskInfo] + all_thinking5b[-1] + all_answer5b[-1], "Sub-task 5b: Make final decision on the correct reactant (A) and diene reactivity sequence (B).", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final answer, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {
        "thinking": thinking5b,
        "answer": answer5b
    }
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5b, answer5b, sub_tasks, agents)
    return final_answer, logs
