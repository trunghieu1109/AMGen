async def forward_185(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Analyze the detailed molecular structure of (1S,4R)-2-vinyl-2-azabicyclo[2.2.1]hept-5-ene, including explicit atom numbering, bond connectivity, and stereochemistry, to identify the key bonds and atoms involved in the Cope rearrangement. Provide a clear atom numbering scheme and highlight the bonds participating in the rearrangement."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing molecular structure with explicit atom numbering and bond connectivity, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Review the Cope rearrangement mechanism focusing on the [3,3]-sigmatropic shift, explicitly identifying the three bonds involved, the required antiperiplanar orbital alignments, and typical transition state conformations (chair vs. boat), especially in rigid bicyclic systems similar to the substrate. Use the structural analysis from Sub-task 1 as context."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, reviewing Cope rearrangement mechanism with focus on bonds and TS conformations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    debate_instruction_3a = "Sub-task 3a: Perform a detailed mechanistic analysis of the Cope rearrangement on the given substrate by explicitly identifying the migrating bonds, possible transition state conformations (chair vs. boat), and orbital interactions, considering the stereochemical constraints imposed by the bicyclic framework and substituents. Include explicit atom numbering and bond details from previous subtasks."
    debate_agents_3a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3a = self.max_round
    all_thinking3a = [[] for _ in range(N_max_3a)]
    all_answer3a = [[] for _ in range(N_max_3a)]
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": debate_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3a):
        for i, agent in enumerate(debate_agents_3a):
            input_infos_3a = [taskInfo, thinking2, answer2]
            if r > 0:
                input_infos_3a.extend(all_thinking3a[r-1])
                input_infos_3a.extend(all_answer3a[r-1])
            thinking3a, answer3a = await agent(input_infos_3a, debate_instruction_3a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, detailed mechanistic analysis, thinking: {thinking3a.content}; answer: {answer3a.content}")
            all_thinking3a[r].append(thinking3a)
            all_answer3a[r].append(answer3a)
    final_decision_agent_3a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3a, answer3a = await final_decision_agent_3a([taskInfo] + all_thinking3a[-1] + all_answer3a[-1], "Sub-task 3a: Make a final decision on the detailed mechanistic analysis of the Cope rearrangement.", is_sub_task=True)
    agents.append(f"Final Decision agent on mechanistic analysis, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    cot_instruction_3b = "Sub-task 3b: Predict the stereochemical outcome of the Cope rearrangement based on the mechanistic analysis from Sub-task 3a, including the configuration of newly formed bonds and ring junctions. Generate plausible product structures with correct stereochemistry."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "CoT"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking3a, answer3a], cot_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, predicting stereochemical outcome, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    debate_instruction_3c = "Sub-task 3c: Validate the predicted product structures by comparing their stereochemical and structural features against the provided multiple-choice options, focusing on hydrogenation patterns, ring saturation, and nomenclature consistency. Use outputs from Sub-task 3b and prior analyses."
    debate_agents_3c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3c = self.max_round
    all_thinking3c = [[] for _ in range(N_max_3c)]
    all_answer3c = [[] for _ in range(N_max_3c)]
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": debate_instruction_3c,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3c):
        for i, agent in enumerate(debate_agents_3c):
            input_infos_3c = [taskInfo, thinking3b, answer3b]
            if r > 0:
                input_infos_3c.extend(all_thinking3c[r-1])
                input_infos_3c.extend(all_answer3c[r-1])
            thinking3c, answer3c = await agent(input_infos_3c, debate_instruction_3c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validating predicted products, thinking: {thinking3c.content}; answer: {answer3c.content}")
            all_thinking3c[r].append(thinking3c)
            all_answer3c[r].append(answer3c)
    final_decision_agent_3c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3c, answer3c = await final_decision_agent_3c([taskInfo] + all_thinking3c[-1] + all_answer3c[-1], "Sub-task 3c: Make a final decision on the validated product structures.", is_sub_task=True)
    agents.append(f"Final Decision agent on product validation, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {
        "thinking": thinking3c,
        "answer": answer3c
    }
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])
    cot_reflect_instruction_4 = "Sub-task 4: Conduct a reflexion step to critically evaluate and reconcile any conflicting mechanistic or stereochemical interpretations from previous subtasks, ensuring the selected product prediction is consistent with textbook Cope rearrangement principles and rigid bicyclic system constraints."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3a, answer3a, thinking3b, answer3b, thinking3c, answer3c]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking and answer of subtask 3a", "thinking and answer of subtask 3b", "thinking and answer of subtask 3c"],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, evaluating and reconciling mechanistic and stereochemical interpretations, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "Please review the mechanistic and stereochemical consistency of the predicted product and provide limitations or corrections.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining product prediction, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction_5 = "Sub-task 5: Select and output the correct multiple-choice answer (A, B, C, or D) corresponding to the validated Cope rearrangement product of (1S,4R)-2-vinyl-2-azabicyclo[2.2.1]hept-5-ene, synthesizing all previous reasoning."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, selecting final multiple-choice answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs