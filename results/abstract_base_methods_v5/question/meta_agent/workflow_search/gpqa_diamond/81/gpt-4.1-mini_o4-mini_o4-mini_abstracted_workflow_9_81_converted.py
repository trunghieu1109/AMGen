async def forward_81(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Analyze the initial reaction between cyclooctatetraene and maleic anhydride in a 1:1 ratio to determine the structure and stereochemistry of product 1, explicitly identifying the Diels–Alder reaction mechanism, regiochemistry, and stereochemical outcome based on known reactivity and molecular orbitals."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing initial reaction, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, analyze the transformation of product 1 upon heating with methanol and a small amount of sulfuric acid to form product 2, identifying the chemical changes such as esterification or ring-opening, and determine the resulting structural and stereochemical modifications."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing transformation of product 1, thinking: {thinking2.content}; answer: {answer2.content}")
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

    cot_instruction_3a = "Sub-task 3a: Perform a detailed conformational analysis of product 2, describing its 3D topology including concave and convex faces, orientation of bulky substituents (e.g., CO2Me groups), and steric environment relevant to the next Diels–Alder reaction. Provide textual 3D descriptions or simple sketches to support reasoning."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, performing conformational analysis, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    cot_instruction_3b = "Sub-task 3b: Evaluate the face-selectivity of the Diels–Alder reaction between product 2 and cyclopentadiene by weighing steric hindrance, secondary orbital interactions, and kinetic versus thermodynamic control to predict endo versus exo approach preferences."
    N = self.max_sc
    cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3b = []
    thinkingmapping_3b = {}
    answermapping_3b = {}
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3b, answer3b = await cot_agents_3b[i]([taskInfo, thinking3a, answer3a], cot_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, evaluating face-selectivity, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_answers_3b.append(answer3b.content)
        thinkingmapping_3b[answer3b.content] = thinking3b
        answermapping_3b[answer3b.content] = answer3b
    most_common_answer_3b = Counter(possible_answers_3b).most_common(1)[0][0]
    thinking3b = thinkingmapping_3b[most_common_answer_3b]
    answer3b = answermapping_3b[most_common_answer_3b]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])

    debate_instruction_3c = "Sub-task 3c: Assign the stereochemistry of the major isomer of product 3 based on the conformational and face-selectivity analyses. Engage in a debate between agents arguing for endo vs exo approaches, considering 3D stereochemical reasoning, substituent orientation, secondary orbital effects, and kinetic vs thermodynamic control. Synthesize arguments to reach a consensus stereochemical assignment with detailed justification."
    debate_agents_3c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3c = self.max_round
    all_thinking3c = [[] for _ in range(N_max_3c)]
    all_answer3c = [[] for _ in range(N_max_3c)]
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": debate_instruction_3c,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3c):
        for i, agent in enumerate(debate_agents_3c):
            input_infos_3c = [taskInfo, thinking3a, answer3a, thinking3b, answer3b]
            if r > 0:
                input_infos_3c.extend(all_thinking3c[r-1])
                input_infos_3c.extend(all_answer3c[r-1])
            thinking3c, answer3c = await agent(input_infos_3c, debate_instruction_3c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating stereochemical assignment, thinking: {thinking3c.content}; answer: {answer3c.content}")
            all_thinking3c[r].append(thinking3c)
            all_answer3c[r].append(answer3c)
    final_decision_agent_3c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3c, answer3c = await final_decision_agent_3c([taskInfo] + all_thinking3c[-1] + all_answer3c[-1], "Sub-task 3c: Make final stereochemical assignment of the major isomer of product 3 based on debate synthesis.", is_sub_task=True)
    agents.append(f"Final Decision agent, stereochemical assignment, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])

    cot_instruction_4 = "Sub-task 4: Compare the predicted stereochemical structure of the major isomer of product 3 with the given multiple-choice options (choices 1 to 4), matching stereochemical descriptors and SMILES strings to identify the correct choice. Include verification checks to ensure consistency between stereochemical assignment and choice matching."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3c, answer3c], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, comparing stereochemical structure with choices, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = "Sub-task 5: Select and output the letter (A, B, C, or D) corresponding to the major isomer of product 3 based on the comparison in Sub-task 4. Engage in a debate among agents to confirm the final choice and resolve any ambiguity."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting final choice, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the major isomer choice letter.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing major isomer choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
