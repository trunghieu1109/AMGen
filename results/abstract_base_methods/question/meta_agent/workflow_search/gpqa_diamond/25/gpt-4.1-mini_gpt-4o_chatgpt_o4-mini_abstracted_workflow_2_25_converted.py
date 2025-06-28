async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = (
        "Sub-task 1a: Perform a detailed structural and mechanistic analysis of the product 8,8-diiodobicyclo[4.2.0]octan-7-one, "
        "including explicit numbering, substitution positions, and functional groups, to understand the key features that must be formed during the reaction."
    )
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, detailed structural and mechanistic analysis of product, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_instruction_1b = (
        "Sub-task 1b: Identify all plausible classes of reactants (A) capable of forming the bicyclo[4.2.0]octan-7-one skeleton with the observed substitution pattern, "
        "considering known reaction mechanisms such as ketene cycloadditions and [2+2] cycloadditions."
    )
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "CoT"
    }
    thinking_1b, answer_1b = await cot_agent_1b([taskInfo, thinking_1a, answer_1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, identifying plausible reactant classes, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_instruction_1c = (
        "Sub-task 1c: Match the plausible reactant classes identified in subtask_1b to the specific candidate reactants (A) provided in the options, "
        "evaluating their structural compatibility and mechanistic feasibility to form the product."
    )
    N_sc_1c = self.max_sc
    cot_agents_1c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1c)]
    possible_answers_1c = []
    thinkingmapping_1c = {}
    answermapping_1c = {}
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_instruction_1c,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1c):
        thinking_1c, answer_1c = await cot_agents_1c[i]([taskInfo, thinking_1b, answer_1b], cot_instruction_1c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1c[i].id}, matching reactant classes to candidates, thinking: {thinking_1c.content}; answer: {answer_1c.content}")
        possible_answers_1c.append(answer_1c.content)
        thinkingmapping_1c[answer_1c.content] = thinking_1c
        answermapping_1c[answer_1c.content] = answer_1c
    answer_1c_content = Counter(possible_answers_1c).most_common(1)[0][0]
    thinking_1c = thinkingmapping_1c[answer_1c_content]
    answer_1c = answermapping_1c[answer_1c_content]
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_desc_1c['response'] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])
    
    debate_instruction_1d = (
        "Sub-task 1d: Conduct a debate-style evaluation where multiple agents independently propose and argue for different reactant candidates and mechanisms "
        "(e.g., ketene vs. cyclobutene pathways), followed by adjudication to select the most mechanistically consistent reactant (A)."
    )
    debate_agents_1d = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1d = self.max_round
    all_thinking_1d = [[] for _ in range(N_max_1d)]
    all_answer_1d = [[] for _ in range(N_max_1d)]
    subtask_desc_1d = {
        "subtask_id": "subtask_1d",
        "instruction": debate_instruction_1d,
        "context": ["user query", "thinking of subtask_1c", "answer of subtask_1c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1d):
        for i, agent in enumerate(debate_agents_1d):
            if r == 0:
                thinking_1d, answer_1d = await agent([taskInfo, thinking_1c, answer_1c], debate_instruction_1d, r, is_sub_task=True)
            else:
                input_infos_1d = [taskInfo, thinking_1c, answer_1c] + all_thinking_1d[r-1] + all_answer_1d[r-1]
                thinking_1d, answer_1d = await agent(input_infos_1d, debate_instruction_1d, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating reactant candidates and mechanisms, thinking: {thinking_1d.content}; answer: {answer_1d.content}")
            all_thinking_1d[r].append(thinking_1d)
            all_answer_1d[r].append(answer_1d)
    final_decision_agent_1d = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1d, answer_1d = await final_decision_agent_1d([taskInfo] + all_thinking_1d[-1] + all_answer_1d[-1], "Sub-task 1d: Make final decision on the most mechanistically consistent reactant (A).", is_sub_task=True)
    agents.append(f"Final Decision agent, adjudicating reactant (A), thinking: {thinking_1d.content}; answer: {answer_1d.content}")
    sub_tasks.append(f"Sub-task 1d output: thinking - {thinking_1d.content}; answer - {answer_1d.content}")
    subtask_desc_1d['response'] = {"thinking": thinking_1d, "answer": answer_1d}
    logs.append(subtask_desc_1d)
    print("Step 1d: ", sub_tasks[-1])
    
    cot_reflect_instruction_1e = (
        "Sub-task 1e: Implement a reflexion checkpoint to re-examine and validate the chosen reactant (A) against product features such as iodine placement and ketone position, "
        "ensuring no inconsistencies before proceeding."
    )
    cot_agent_1e = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1e = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1e = self.max_round
    cot_inputs_1e = [taskInfo, thinking_1d, answer_1d]
    subtask_desc_1e = {
        "subtask_id": "subtask_1e",
        "instruction": cot_reflect_instruction_1e,
        "context": ["user query", "thinking of subtask_1d", "answer of subtask_1d"],
        "agent_collaboration": "Reflexion"
    }
    thinking_1e, answer_1e = await cot_agent_1e(cot_inputs_1e, cot_reflect_instruction_1e, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1e.id}, validating chosen reactant (A), thinking: {thinking_1e.content}; answer: {answer_1e.content}")
    for i in range(N_max_1e):
        feedback_1e, correct_1e = await critic_agent_1e([taskInfo, thinking_1e, answer_1e],
                                                      "Please review the chosen reactant (A) validation and provide any inconsistencies or limitations.",
                                                      i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1e.id}, providing feedback, thinking: {feedback_1e.content}; answer: {correct_1e.content}")
        if correct_1e.content == "True":
            break
        cot_inputs_1e.extend([thinking_1e, answer_1e, feedback_1e])
        thinking_1e, answer_1e = await cot_agent_1e(cot_inputs_1e, cot_reflect_instruction_1e, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1e.id}, refining reactant (A) validation, thinking: {thinking_1e.content}; answer: {answer_1e.content}")
    sub_tasks.append(f"Sub-task 1e output: thinking - {thinking_1e.content}; answer - {answer_1e.content}")
    subtask_desc_1e['response'] = {"thinking": thinking_1e, "answer": answer_1e}
    logs.append(subtask_desc_1e)
    print("Step 1e: ", sub_tasks[-1])
    
    cot_instruction_2a = (
        "Sub-task 2a: Analyze the four given dienes (B) — 2,3-dimethylbuta-1,3-diene, (2E,4E)-hexa-2,4-diene, cyclopenta-1,3-diene, and (2Z,4Z)-hexa-2,4-diene — "
        "to determine their relative reactivity order based on conjugation, substitution, stereochemistry, and electronic effects."
    )
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_2a, answer_2a = await cot_agent_2a([taskInfo], cot_instruction_2a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2a.id}, analyzing dienes reactivity, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_instruction_2b = (
        "Sub-task 2b: Establish the correct sequence of the dienes from most reactive to least reactive by comparing and integrating the electronic and structural features derived in subtask_2a."
    )
    N_sc_2b = self.max_sc
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2b)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask_2a", "answer of subtask_2a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2b):
        thinking_2b, answer_2b = await cot_agents_2b[i]([taskInfo, thinking_2a, answer_2a], cot_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, determining dienes reactivity order, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
        possible_answers_2b.append(answer_2b.content)
        thinkingmapping_2b[answer_2b.content] = thinking_2b
        answermapping_2b[answer_2b.content] = answer_2b
    answer_2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking_2b = thinkingmapping_2b[answer_2b_content]
    answer_2b = answermapping_2b[answer_2b_content]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_instruction_3 = (
        "Sub-task 3: Integrate the validated reactant (A) from subtask_1e and the diene reactivity order from subtask_2b, "
        "applying a self-consistency chain-of-thought approach to cross-validate and ensure consistency between both findings."
    )
    N_sc_3 = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask_1e", "answer of subtask_1e", "thinking of subtask_2b", "answer of subtask_2b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3):
        thinking_3, answer_3 = await cot_agents_3[i]([taskInfo, thinking_1e, answer_1e, thinking_2b, answer_2b], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, cross-validating reactant and diene reactivity, thinking: {thinking_3.content}; answer: {answer_3.content}")
        possible_answers_3.append(answer_3.content)
        thinkingmapping_3[answer_3.content] = thinking_3
        answermapping_3[answer_3.content] = answer_3
    answer_3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking_3 = thinkingmapping_3[answer_3_content]
    answer_3 = answermapping_3[answer_3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])
    
    debate_instruction_4 = (
        "Sub-task 4: Select the correct multiple-choice answer (A, B, C, or D) that matches both the identified reactant (A) and the correct reactivity sequence of dienes (B), "
        "ensuring the final answer is consistent with all mechanistic and structural analyses."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask_3", "answer of subtask_3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking_4, answer_4 = await agent([taskInfo, thinking_3, answer_3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking_3, answer_3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking_4, answer_4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting final multiple-choice answer, thinking: {thinking_4.content}; answer: {answer_4.content}")
            all_thinking_4[r].append(thinking_4)
            all_answer_4[r].append(answer_4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 4: Make final decision on the correct multiple-choice answer matching reactant (A) and dienes reactivity order.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final answer, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_4, answer_4, sub_tasks, agents)
    return final_answer, logs
