async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = "Sub-task 1a: Identify and list all plausible reaction mechanisms that could lead to the formation of 8,8-diiodobicyclo[4.2.0]octan-7-one from cyclohexene and a reactant A, including ketene [2+2] cycloaddition, Diels–Alder [4+2] cycloaddition, and radical pathways. Consider mechanistic feasibility and chemical principles."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, listing plausible reaction mechanisms, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_instruction_1b = "Sub-task 1b: Analyze the origin and role of iodine atoms in the product structure to infer the nature and identity of the reactant A, focusing on how iodine incorporation occurs mechanistically. Use insights from Sub-task 1a."
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "CoT"
    }
    thinking_1b, answer_1b = await cot_agent_1b([taskInfo, thinking_1a, answer_1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, analyzing iodine origin and role, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_instruction_1c = "Sub-task 1c: Match each candidate reactant A from the given options to the product structure by evaluating mechanistic feasibility and consistency with the iodine incorporation and reaction type identified in previous subtasks. Use self-consistency to generate multiple hypotheses and select the most plausible reactant."
    N = self.max_sc
    cot_agents_1c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1c = []
    thinkingmapping_1c = {}
    answermapping_1c = {}
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_instruction_1c,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_1c, answer_1c = await cot_agents_1c[i]([taskInfo, thinking_1b, answer_1b], cot_instruction_1c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1c[i].id}, evaluating candidate reactants, thinking: {thinking_1c.content}; answer: {answer_1c.content}")
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
    
    cot_reflect_instruction_1d = "Sub-task 1d: Perform a reflexive review of the assumptions and conclusions from Sub-tasks 1a to 1c, critically evaluating alternative mechanisms and reactants, referencing chemical literature or known precedents to confirm or revise the selected reactant and mechanism."
    cot_agent_1d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_1d = [taskInfo, thinking_1a, answer_1a, thinking_1b, answer_1b, thinking_1c, answer_1c]
    subtask_desc_1d = {
        "subtask_id": "subtask_1d",
        "instruction": cot_reflect_instruction_1d,
        "context": ["user query", "thinking and answer of subtask_1a", "thinking and answer of subtask_1b", "thinking and answer of subtask_1c"],
        "agent_collaboration": "Reflexion"
    }
    thinking_1d, answer_1d = await cot_agent_1d(cot_inputs_1d, cot_reflect_instruction_1d, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1d.id}, reflexive review of reactant and mechanism, thinking: {thinking_1d.content}; answer: {answer_1d.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_1d([taskInfo, thinking_1d, answer_1d], "Critically evaluate the assumptions and conclusions about the reactant and mechanism, provide limitations or confirm correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1d.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1d.extend([thinking_1d, answer_1d, feedback])
        thinking_1d, answer_1d = await cot_agent_1d(cot_inputs_1d, cot_reflect_instruction_1d, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1d.id}, refining reflexive review, thinking: {thinking_1d.content}; answer: {answer_1d.content}")
    sub_tasks.append(f"Sub-task 1d output: thinking - {thinking_1d.content}; answer - {answer_1d.content}")
    subtask_desc_1d['response'] = {"thinking": thinking_1d, "answer": answer_1d}
    logs.append(subtask_desc_1d)
    print("Step 1d: ", sub_tasks[-1])
    
    cot_instruction_2 = "Sub-task 2: Analyze the four given diene options (B) to understand their structural features, conjugation, substitution, and stereochemistry that influence their reactivity in typical diene reactions. Provide detailed reasoning."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_2, answer_2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing diene options, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_3 = "Sub-task 3: Assess the reactivity of the given dienes (B) based on their conjugation, substitution pattern, and stereochemistry to establish a reactivity order from most reactive to least reactive, supported by chemical principles. Use self-consistency to ensure robustness."
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask_2", "answer of subtask_2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_3, answer_3 = await cot_agents_3[i]([taskInfo, thinking_2, answer_2], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, assessing dienes reactivity, thinking: {thinking_3.content}; answer: {answer_3.content}")
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
    
    debate_instruction_4 = "Sub-task 4: Integrate the validated reactant identification from Sub-task 1d with the diene reactivity order from Sub-task 3 to select the correct multiple-choice option that accurately identifies reactant A and the reactivity sequence of dienes B. Use a debate pattern to ensure robustness."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask_1d", "answer of subtask_1d", "thinking of subtask_3", "answer of subtask_3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking_4, answer_4 = await agent([taskInfo, thinking_1d, answer_1d, thinking_3, answer_3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking_1d, answer_1d, thinking_3, answer_3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking_4, answer_4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating reactant and diene reactivity, thinking: {thinking_4.content}; answer: {answer_4.content}")
            all_thinking_4[r].append(thinking_4)
            all_answer_4[r].append(answer_4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 4: Make final decision on the correct reactant (A) and diene reactivity sequence (B).", is_sub_task=True)
    agents.append(f"Final Decision agent, making final selection, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_4, answer_4, sub_tasks, agents)
    return final_answer, logs