async def forward_81(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_debate_instruction_1a = "Sub-task 1a: Perform a detailed conformational analysis of cyclooctatetraene (COT) to determine its 3D shape, reactive diene moieties, and how its tub-shaped conformation influences its reactivity with maleic anhydride. Generate multiple plausible conformations and discuss their impact."
    debate_agents_1a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1a = self.max_round
    all_thinking_1a = [[] for _ in range(N_max_1a)]
    all_answer_1a = [[] for _ in range(N_max_1a)]
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_debate_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1a):
        for i, agent in enumerate(debate_agents_1a):
            if r == 0:
                thinking_1a, answer_1a = await agent([taskInfo], cot_debate_instruction_1a, r, is_sub_task=True)
            else:
                input_infos_1a = [taskInfo] + all_thinking_1a[r-1] + all_answer_1a[r-1]
                thinking_1a, answer_1a = await agent(input_infos_1a, cot_debate_instruction_1a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing COT conformation, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
            all_thinking_1a[r].append(thinking_1a)
            all_answer_1a[r].append(answer_1a)
    final_decision_agent_1a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1a, answer_1a = await final_decision_agent_1a([taskInfo] + all_thinking_1a[-1] + all_answer_1a[-1], "Sub-task 1a: Make final decision on the most plausible COT conformation and reactive diene moieties.", is_sub_task=True)
    agents.append(f"Final Decision agent on COT conformation, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a["response"] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])

    cot_debate_instruction_1b = "Sub-task 1b: Analyze possible reaction pathways between COT and maleic anhydride, including endo vs exo approaches and potential stepwise mechanisms, considering the nonplanar conformation of COT. Debate the mechanistic plausibility and stereochemical implications."
    debate_agents_1b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1b = self.max_round
    all_thinking_1b = [[] for _ in range(N_max_1b)]
    all_answer_1b = [[] for _ in range(N_max_1b)]
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_debate_instruction_1b,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1b):
        for i, agent in enumerate(debate_agents_1b):
            if r == 0:
                thinking_1b, answer_1b = await agent([taskInfo, thinking_1a, answer_1a], cot_debate_instruction_1b, r, is_sub_task=True)
            else:
                input_infos_1b = [taskInfo, thinking_1a, answer_1a] + all_thinking_1b[r-1] + all_answer_1b[r-1]
                thinking_1b, answer_1b = await agent(input_infos_1b, cot_debate_instruction_1b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing reaction pathways, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
            all_thinking_1b[r].append(thinking_1b)
            all_answer_1b[r].append(answer_1b)
    final_decision_agent_1b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1b, answer_1b = await final_decision_agent_1b([taskInfo, thinking_1a, answer_1a] + all_thinking_1b[-1] + all_answer_1b[-1], "Sub-task 1b: Make final decision on the most plausible reaction pathway and stereochemical implications.", is_sub_task=True)
    agents.append(f"Final Decision agent on reaction pathways, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b["response"] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])

    cot_sc_instruction_1c = "Sub-task 1c: Predict the stereochemical outcome of the Diels–Alder reaction between COT and maleic anhydride, enumerating all plausible stereoisomers of product 1 and their relative stabilities based on steric, electronic, and conformational factors. Use self-consistency to generate multiple plausible outcomes."
    N_1c = self.max_sc
    cot_agents_1c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_1c)]
    possible_answers_1c = []
    thinkingmapping_1c = {}
    answermapping_1c = {}
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_sc_instruction_1c,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_1c):
        thinking_1c, answer_1c = await cot_agents_1c[i]([taskInfo, thinking_1a, answer_1a, thinking_1b, answer_1b], cot_sc_instruction_1c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1c[i].id}, predicting stereochemical outcomes of product 1, thinking: {thinking_1c.content}; answer: {answer_1c.content}")
        possible_answers_1c.append(answer_1c.content)
        thinkingmapping_1c[answer_1c.content] = thinking_1c
        answermapping_1c[answer_1c.content] = answer_1c
    answer_1c_content = Counter(possible_answers_1c).most_common(1)[0][0]
    thinking_1c = thinkingmapping_1c[answer_1c_content]
    answer_1c = answermapping_1c[answer_1c_content]
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_desc_1c["response"] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])

    cot_instruction_1d = "Sub-task 1d: Map and assign stereochemical configurations (R/S) to all new stereocenters formed in product 1, preparing a stereochemical table for comparison with downstream products."
    cot_agent_1d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1d = {
        "subtask_id": "subtask_1d",
        "instruction": cot_instruction_1d,
        "context": ["user query", "thinking of subtask 1c", "answer of subtask 1c"],
        "agent_collaboration": "CoT"
    }
    thinking_1d, answer_1d = await cot_agent_1d([taskInfo, thinking_1c, answer_1c], cot_instruction_1d, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1d.id}, mapping stereochemistry of product 1, thinking: {thinking_1d.content}; answer: {answer_1d.content}")
    sub_tasks.append(f"Sub-task 1d output: thinking - {thinking_1d.content}; answer - {answer_1d.content}")
    subtask_desc_1d["response"] = {"thinking": thinking_1d, "answer": answer_1d}
    logs.append(subtask_desc_1d)
    print("Step 1d: ", sub_tasks[-1])

    cot_sc_instruction_2a = "Sub-task 2a: Determine the chemical transformation occurring when product 1 is heated with methanol and sulfuric acid, including ring opening or esterification, and identify any changes in stereochemistry or ring strain. Use self-consistency to consider multiple mechanistic possibilities."
    N_2a = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_2a)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", "thinking of subtask 1d", "answer of subtask 1d"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_2a):
        thinking_2a, answer_2a = await cot_agents_2a[i]([taskInfo, thinking_1d, answer_1d], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, determining chemical transformation of product 1 to 2, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
        possible_answers_2a.append(answer_2a.content)
        thinkingmapping_2a[answer_2a.content] = thinking_2a
        answermapping_2a[answer_2a.content] = answer_2a
    answer_2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking_2a = thinkingmapping_2a[answer_2a_content]
    answer_2a = answermapping_2a[answer_2a_content]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a["response"] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])

    cot_instruction_2b = "Sub-task 2b: Enumerate and assign stereochemical configurations of product 2, considering possible regio- and stereoisomers formed during the methanolysis step, and prepare a stereochemical mapping table."
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "CoT"
    }
    thinking_2b, answer_2b = await cot_agent_2b([taskInfo, thinking_2a, answer_2a], cot_instruction_2b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2b.id}, mapping stereochemistry of product 2, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b["response"] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])

    cot_sc_instruction_3a = "Sub-task 3a: Identify the reactive dienophile site(s) on product 2 that will participate in the Diels–Alder reaction with cyclopentadiene, considering electronic and steric factors. Use self-consistency to generate multiple hypotheses."
    N_3a = self.max_sc
    cot_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_3a)]
    possible_answers_3a = []
    thinkingmapping_3a = {}
    answermapping_3a = {}
    subtask_desc_3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_3a):
        thinking_3a, answer_3a = await cot_agents_3a[i]([taskInfo, thinking_2b, answer_2b], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, identifying reactive dienophile sites on product 2, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
        possible_answers_3a.append(answer_3a.content)
        thinkingmapping_3a[answer_3a.content] = thinking_3a
        answermapping_3a[answer_3a.content] = answer_3a
    answer_3a_content = Counter(possible_answers_3a).most_common(1)[0][0]
    thinking_3a = thinkingmapping_3a[answer_3a_content]
    answer_3a = answermapping_3a[answer_3a_content]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking_3a.content}; answer - {answer_3a.content}")
    subtask_desc_3a["response"] = {"thinking": thinking_3a, "answer": answer_3a}
    logs.append(subtask_desc_3a)
    print("Step 3a: ", sub_tasks[-1])

    cot_sc_instruction_3b = "Sub-task 3b: Perform a detailed regio- and stereochemical analysis of the Diels–Alder reaction between product 2 and cyclopentadiene, enumerating all plausible stereoisomers of product 3, including endo/exo selectivity and stereochemical configurations at all new stereocenters. Use self-consistency to generate multiple plausible outcomes."
    N_3b = self.max_sc
    cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_3b)]
    possible_answers_3b = []
    thinkingmapping_3b = {}
    answermapping_3b = {}
    subtask_desc_3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_3b):
        thinking_3b, answer_3b = await cot_agents_3b[i]([taskInfo, thinking_3a, answer_3a], cot_sc_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, analyzing regio- and stereochemistry of product 3, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
        possible_answers_3b.append(answer_3b.content)
        thinkingmapping_3b[answer_3b.content] = thinking_3b
        answermapping_3b[answer_3b.content] = answer_3b
    answer_3b_content = Counter(possible_answers_3b).most_common(1)[0][0]
    thinking_3b = thinkingmapping_3b[answer_3b_content]
    answer_3b = answermapping_3b[answer_3b_content]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking_3b.content}; answer - {answer_3b.content}")
    subtask_desc_3b["response"] = {"thinking": thinking_3b, "answer": answer_3b}
    logs.append(subtask_desc_3b)
    print("Step 3b: ", sub_tasks[-1])

    cot_instruction_3c = "Sub-task 3c: Generate a comprehensive stereochemical mapping table for all plausible isomers of product 3, assigning R/S configurations and comparing their relative stabilities based on steric, electronic, and conformational considerations."
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_instruction_3c,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "CoT"
    }
    thinking_3c, answer_3c = await cot_agent_3c([taskInfo, thinking_3b, answer_3b], cot_instruction_3c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3c.id}, mapping stereochemistry of product 3 isomers, thinking: {thinking_3c.content}; answer: {answer_3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking_3c.content}; answer - {answer_3c.content}")
    subtask_desc_3c["response"] = {"thinking": thinking_3c, "answer": answer_3c}
    logs.append(subtask_desc_3c)
    print("Step 3c: ", sub_tasks[-1])

    cot_reflect_instruction_4a = "Sub-task 4a: Cross-validate the stereochemical assignments of product 3 isomers by applying CIP rules, atomic numbering, and alternative stereochemical assignment methods to ensure accuracy."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4a = self.max_round
    cot_inputs_4a = [taskInfo, thinking_3c, answer_3c]
    subtask_desc_4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_reflect_instruction_4a,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "Reflexion"
    }
    thinking_4a, answer_4a = await cot_agent_4a(cot_inputs_4a, cot_reflect_instruction_4a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, cross-validating stereochemical assignments, thinking: {thinking_4a.content}; answer: {answer_4a.content}")
    for i in range(N_max_4a):
        feedback_4a, correct_4a = await critic_agent_4a([taskInfo, thinking_4a, answer_4a], "please review the stereochemical assignments validation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4a.id}, providing feedback, thinking: {feedback_4a.content}; answer: {correct_4a.content}")
        if correct_4a.content == "True":
            break
        cot_inputs_4a.extend([thinking_4a, answer_4a, feedback_4a])
        thinking_4a, answer_4a = await cot_agent_4a(cot_inputs_4a, cot_reflect_instruction_4a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, refining stereochemical validation, thinking: {thinking_4a.content}; answer: {answer_4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking_4a.content}; answer - {answer_4a.content}")
    subtask_desc_4a["response"] = {"thinking": thinking_4a, "answer": answer_4a}
    logs.append(subtask_desc_4a)
    print("Step 4a: ", sub_tasks[-1])

    debate_instruction_4b = "Sub-task 4b: Compare the stereochemical data of all plausible product 3 isomers against the given multiple-choice options, systematically ruling out inconsistent candidates based on stereochemical discrepancies and mechanistic reasoning."
    debate_agents_4b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4b = self.max_round
    all_thinking_4b = [[] for _ in range(N_max_4b)]
    all_answer_4b = [[] for _ in range(N_max_4b)]
    subtask_desc_4b = {
        "subtask_id": "subtask_4b",
        "instruction": debate_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4b):
        for i, agent in enumerate(debate_agents_4b):
            if r == 0:
                thinking_4b, answer_4b = await agent([taskInfo, thinking_4a, answer_4a], debate_instruction_4b, r, is_sub_task=True)
            else:
                input_infos_4b = [taskInfo, thinking_4a, answer_4a] + all_thinking_4b[r-1] + all_answer_4b[r-1]
                thinking_4b, answer_4b = await agent(input_infos_4b, debate_instruction_4b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing stereochemical data with choices, thinking: {thinking_4b.content}; answer: {answer_4b.content}")
            all_thinking_4b[r].append(thinking_4b)
            all_answer_4b[r].append(answer_4b)
    final_decision_agent_4b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4b, answer_4b = await final_decision_agent_4b([taskInfo] + all_thinking_4b[-1] + all_answer_4b[-1], "Sub-task 4b: Make final decision on ruling out inconsistent isomers.", is_sub_task=True)
    agents.append(f"Final Decision agent on stereochemical comparison, thinking: {thinking_4b.content}; answer: {answer_4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking_4b.content}; answer - {answer_4b.content}")
    subtask_desc_4b["response"] = {"thinking": thinking_4b, "answer": answer_4b}
    logs.append(subtask_desc_4b)
    print("Step 4b: ", sub_tasks[-1])

    cot_instruction_4c = "Sub-task 4c: Select and justify the major isomer of product 3 based on the comprehensive stereochemical analysis, mechanistic plausibility, and comparison with the provided choices."
    cot_agent_4c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_instruction_4c,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "CoT"
    }
    thinking_4c, answer_4c = await cot_agent_4c([taskInfo, thinking_4b, answer_4b], cot_instruction_4c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4c.id}, selecting and justifying major isomer, thinking: {thinking_4c.content}; answer: {answer_4c.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking_4c.content}; answer - {answer_4c.content}")
    subtask_desc_4c["response"] = {"thinking": thinking_4c, "answer": answer_4c}
    logs.append(subtask_desc_4c)
    print("Step 4c: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_4c, answer_4c, sub_tasks, agents)
    return final_answer, logs
