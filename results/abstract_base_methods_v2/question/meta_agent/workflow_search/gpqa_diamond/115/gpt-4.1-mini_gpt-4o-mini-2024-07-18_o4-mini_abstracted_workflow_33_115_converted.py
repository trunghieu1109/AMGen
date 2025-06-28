async def forward_115(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1a = "Sub-task 1a: Analyze the proton NMR integration data of compound A to determine the number of equivalent protons and infer possible symmetry elements, focusing on the 6H triplet at 0.9 ppm and 4H quartet at 1.3 ppm."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, analyzing NMR integration and symmetry, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a["response"] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])

    cot_instruction_1b = "Sub-task 1b: Interpret the splitting patterns (triplet and quartet) using the n+1 rule to deduce the number and type of neighboring protons for each signal in compound A, with context from Sub-task 1a."
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "CoT"
    }
    thinking_1b, answer_1b = await cot_agent_1b([taskInfo, thinking_1a, answer_1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, interpreting splitting patterns, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b["response"] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])

    cot_instruction_1c = "Sub-task 1c: Generate all plausible structural fragments and candidate structures for compound A consistent with the NMR data, considering symmetry, equivalent groups, and splitting patterns, including linear and branched alkane possibilities, with context from Sub-task 1b."
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_instruction_1c,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "CoT"
    }
    thinking_1c, answer_1c = await cot_agent_1c([taskInfo, thinking_1b, answer_1b], cot_instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, generating candidate structures, thinking: {thinking_1c.content}; answer: {answer_1c.content}")
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_desc_1c["response"] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])

    cot_instruction_1d = "Sub-task 1d: Perform a self-consistency check and debate among candidate structures generated in Sub-task 1c, critically comparing each against the NMR data to select the most consistent structure for compound A."
    N_sc_1d = self.max_sc
    debate_agents_1d = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_1d = [[] for _ in range(N_sc_1d)]
    all_answer_1d = [[] for _ in range(N_sc_1d)]
    subtask_desc_1d = {
        "subtask_id": "subtask_1d",
        "instruction": cot_instruction_1d,
        "context": ["user query", "thinking of subtask_1c", "answer of subtask_1c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_sc_1d):
        for i, agent in enumerate(debate_agents_1d):
            if r == 0:
                thinking_1d, answer_1d = await agent([taskInfo, thinking_1c, answer_1c], cot_instruction_1d, r, is_sub_task=True)
            else:
                input_infos_1d = [taskInfo, thinking_1c, answer_1c] + all_thinking_1d[r-1] + all_answer_1d[r-1]
                thinking_1d, answer_1d = await agent(input_infos_1d, cot_instruction_1d, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating candidate structures, thinking: {thinking_1d.content}; answer: {answer_1d.content}")
            all_thinking_1d[r].append(thinking_1d)
            all_answer_1d[r].append(answer_1d)
    final_decision_agent_1d = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1d, answer_1d = await final_decision_agent_1d([taskInfo] + all_thinking_1d[-1] + all_answer_1d[-1], "Sub-task 1d: Make final decision on the most consistent structure for compound A.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting structure of compound A, thinking: {thinking_1d.content}; answer: {answer_1d.content}")
    sub_tasks.append(f"Sub-task 1d output: thinking - {thinking_1d.content}; answer - {answer_1d.content}")
    subtask_desc_1d["response"] = {"thinking": thinking_1d, "answer": answer_1d}
    logs.append(subtask_desc_1d)
    print("Step 1d: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Based on the confirmed structure of compound A from Sub-task 1d, determine the most likely site of monobromination to form compound B, considering reactivity and structural features."
    N_sc_2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask_1d", "answer of subtask_1d"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2):
        thinking_2, answer_2 = await cot_agents_2[i]([taskInfo, thinking_1d, answer_1d], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining bromination site, thinking: {thinking_2.content}; answer: {answer_2.content}")
        possible_answers_2.append(answer_2.content)
        thinkingmapping_2[answer_2.content] = thinking_2
        answermapping_2[answer_2.content] = answer_2
    answer_2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking_2 = thinkingmapping_2[answer_2_content]
    answer_2 = answermapping_2[answer_2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2["response"] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3a = "Sub-task 3a: Analyze the reaction of compound B with alcoholic KOH to deduce the elimination product(s) formed (compound C), including the identification of possible geometrical (cis/trans) isomers, with context from Sub-task 2."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask_2", "answer of subtask_2"],
        "agent_collaboration": "CoT"
    }
    thinking_3a, answer_3a = await cot_agent_3a([taskInfo, thinking_2, answer_2], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, analyzing elimination reaction, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking_3a.content}; answer - {answer_3a.content}")
    subtask_desc_3a["response"] = {"thinking": thinking_3a, "answer": answer_3a}
    logs.append(subtask_desc_3a)
    print("Step 3a: ", sub_tasks[-1])

    cot_reflect_instruction_3b = "Sub-task 3b: Perform a reflexion step to reassess the structural assignments of compound A and B in light of the elimination product(s) and their stereochemistry, ensuring consistency across all intermediates."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3b = self.max_round
    cot_inputs_3b = [taskInfo, thinking_1d, answer_1d, thinking_2, answer_2, thinking_3a, answer_3a]
    subtask_desc_3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask_1d", "answer of subtask_1d", "thinking of subtask_2", "answer of subtask_2", "thinking of subtask_3a", "answer of subtask_3a"],
        "agent_collaboration": "Reflexion"
    }
    thinking_3b, answer_3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, reassessing structures A and B, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
    for i in range(N_max_3b):
        feedback_3b, correct_3b = await critic_agent_3b([taskInfo, thinking_3b, answer_3b], "please review the reassessment of compounds A and B for consistency with elimination product(s) and stereochemistry, and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, providing feedback, thinking: {feedback_3b.content}; answer: {correct_3b.content}")
        if correct_3b.content == "True":
            break
        cot_inputs_3b.extend([thinking_3b, answer_3b, feedback_3b])
        thinking_3b, answer_3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining reassessment, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking_3b.content}; answer - {answer_3b.content}")
    subtask_desc_3b["response"] = {"thinking": thinking_3b, "answer": answer_3b}
    logs.append(subtask_desc_3b)
    print("Step 3b: ", sub_tasks[-1])

    cot_instruction_4a = "Sub-task 4a: Identify the structure and stereochemistry of the cis-isomer of compound C, detailing its configuration and relevant stereochemical descriptors, with context from Sub-task 3b."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking of subtask_3b", "answer of subtask_3b"],
        "agent_collaboration": "CoT"
    }
    thinking_4a, answer_4a = await cot_agent_4a([taskInfo, thinking_3b, answer_3b], cot_instruction_4a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4a.id}, identifying cis-isomer structure and stereochemistry, thinking: {thinking_4a.content}; answer: {answer_4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking_4a.content}; answer - {answer_4a.content}")
    subtask_desc_4a["response"] = {"thinking": thinking_4a, "answer": answer_4a}
    logs.append(subtask_desc_4a)
    print("Step 4a: ", sub_tasks[-1])

    cot_instruction_4b = "Sub-task 4b: Analyze the reaction mechanism between the cis-isomer of compound C and (1E,3E)-penta-1,3-dien-1-ol, determining the type of reaction and the structural features of the resulting compound D, with context from Sub-task 4a."
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4b = self.max_round
    cot_inputs_4b = [taskInfo, thinking_4a, answer_4a]
    subtask_desc_4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_instruction_4b,
        "context": ["user query", "thinking of subtask_4a", "answer of subtask_4a"],
        "agent_collaboration": "Reflexion"
    }
    thinking_4b, answer_4b = await cot_agent_4b(cot_inputs_4b, cot_instruction_4b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, analyzing reaction mechanism to form compound D, thinking: {thinking_4b.content}; answer: {answer_4b.content}")
    for i in range(N_max_4b):
        feedback_4b, correct_4b = await critic_agent_4b([taskInfo, thinking_4b, answer_4b], "please review the reaction mechanism and structural features of compound D, providing limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4b.id}, providing feedback, thinking: {feedback_4b.content}; answer: {correct_4b.content}")
        if correct_4b.content == "True":
            break
        cot_inputs_4b.extend([thinking_4b, answer_4b, feedback_4b])
        thinking_4b, answer_4b = await cot_agent_4b(cot_inputs_4b, cot_instruction_4b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, refining reaction mechanism analysis, thinking: {thinking_4b.content}; answer: {answer_4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking_4b.content}; answer - {answer_4b.content}")
    subtask_desc_4b["response"] = {"thinking": thinking_4b, "answer": answer_4b}
    logs.append(subtask_desc_4b)
    print("Step 4b: ", sub_tasks[-1])

    debate_instruction_5a = "Sub-task 5a: Compare the deduced structure and stereochemistry of compound D with the given multiple-choice options, evaluating substitution patterns and stereochemical configurations, with context from Sub-task 4b."
    debate_agents_5a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5a = self.max_round
    all_thinking_5a = [[] for _ in range(N_max_5a)]
    all_answer_5a = [[] for _ in range(N_max_5a)]
    subtask_desc_5a = {
        "subtask_id": "subtask_5a",
        "instruction": debate_instruction_5a,
        "context": ["user query", "thinking of subtask_4b", "answer of subtask_4b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5a):
        for i, agent in enumerate(debate_agents_5a):
            if r == 0:
                thinking_5a, answer_5a = await agent([taskInfo, thinking_4b, answer_4b], debate_instruction_5a, r, is_sub_task=True)
            else:
                input_infos_5a = [taskInfo, thinking_4b, answer_4b] + all_thinking_5a[r-1] + all_answer_5a[r-1]
                thinking_5a, answer_5a = await agent(input_infos_5a, debate_instruction_5a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing compound D with options, thinking: {thinking_5a.content}; answer: {answer_5a.content}")
            all_thinking_5a[r].append(thinking_5a)
            all_answer_5a[r].append(answer_5a)
    final_decision_agent_5b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5b, answer_5b = await final_decision_agent_5b([taskInfo] + all_thinking_5a[-1] + all_answer_5a[-1], "Sub-task 5b: Conduct a final reflexion to verify overall consistency and correctness of the reasoning chain before selecting the final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, verifying overall consistency and selecting final answer, thinking: {thinking_5b.content}; answer: {answer_5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking_5b.content}; answer - {answer_5b.content}")
    subtask_desc_5b = {
        "subtask_id": "subtask_5b",
        "instruction": "Sub-task 5b: Conduct a final reflexion to verify overall consistency and correctness of the reasoning chain before selecting the final answer.",
        "context": ["user query", "thinking of subtask_5a", "answer of subtask_5a"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc_5b["response"] = {"thinking": thinking_5b, "answer": answer_5b}
    logs.append(subtask_desc_5b)
    print("Step 5b: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_5b, answer_5b, sub_tasks, agents)
    return final_answer, logs
