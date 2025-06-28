async def forward_32(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Identify and characterize the reactants 2,5-dimethylthiophene and furan-2,5-dione, focusing on their structural features relevant to the [4+2] cycloaddition reaction."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying and characterizing reactants, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_instruction_2 = "Sub-task 2: Analyze the reaction conditions (heat) and the nature of the [4+2] cycloaddition to determine the expected reaction mechanism and possible stereochemical outcomes (endo vs exo), based on the characterization of reactants from Sub-task 1."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing reaction conditions and mechanism, thinking: {thinking2.content}; answer: {answer2.content}")
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
    subtask_3a_instruction = "Sub-task 3a: Determine and justify the bicyclic core structure formed in the cycloaddition product by distinguishing between the possible frameworks (epoxybenzo[c]thiophene vs epithioisobenzofuran). Provide detailed chemical reasoning supported by structural diagrams, literature precedents, or spectral data to confirm the correct bicyclic core."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": subtask_3a_instruction,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2], subtask_3a_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, determining bicyclic core structure, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a["response"] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    subtask_3b_instruction = "Sub-task 3b: Assign the stereochemistry of the exo product based on the confirmed bicyclic core structure from Subtask 3a. Apply CIP priority rules step-by-step to each stereocenter, explicitly listing priorities and justifications, and generate a 3D stereochemical representation or equivalent to support the assignment."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": subtask_3b_instruction,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "CoT"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking3a, answer3a], subtask_3b_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, assigning stereochemistry with CIP rules, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b["response"] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    subtask_3c_instruction = "Sub-task 3c: Conduct a Self-Consistency Chain-of-Thought (SC-CoT) analysis by having multiple independent agents perform CIP stereochemical assignments and compare results to reach a consensus on the stereochemical configuration, reducing individual bias or error."
    N = self.max_sc
    cot_agents_3c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3c = []
    thinkingmapping_3c = {}
    answermapping_3c = {}
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": subtask_3c_instruction,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3c, answer3c = await cot_agents_3c[i]([taskInfo, thinking3b, answer3b], subtask_3c_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3c[i].id}, performing CIP stereochemical assignment, thinking: {thinking3c.content}; answer: {answer3c.content}")
        possible_answers_3c.append(answer3c.content)
        thinkingmapping_3c[answer3c.content] = thinking3c
        answermapping_3c[answer3c.content] = answer3c
    most_common_answer_3c = Counter(possible_answers_3c).most_common(1)[0][0]
    thinking3c = thinkingmapping_3c[most_common_answer_3c]
    answer3c = answermapping_3c[most_common_answer_3c]
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c["response"] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])
    subtask_3d_instruction = "Sub-task 3d: Perform a reflexive cross-validation of the bicyclic core identity and stereochemical assignments from Subtasks 3a–3c. Critically review and challenge majority assumptions, compare results against known chemical references, computational stereochemistry tools, or databases, and identify any inconsistencies or errors."
    cot_agent_3d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3d = self.max_round
    cot_inputs_3d = [taskInfo, thinking3a, answer3a, thinking3b, answer3b, thinking3c, answer3c]
    subtask_desc3d = {
        "subtask_id": "subtask_3d",
        "instruction": subtask_3d_instruction,
        "context": ["user query", "thinking and answer of subtasks 3a, 3b, 3c"],
        "agent_collaboration": "Reflexion"
    }
    thinking3d, answer3d = await cot_agent_3d(cot_inputs_3d, subtask_3d_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3d.id}, cross-validating bicyclic core and stereochemistry, thinking: {thinking3d.content}; answer: {answer3d.content}")
    for i in range(N_max_3d):
        feedback, correct = await critic_agent_3d([taskInfo, thinking3d, answer3d], "Please review the bicyclic core and stereochemical assignments and provide limitations or corrections.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3d.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_3d.extend([thinking3d, answer3d, feedback])
        thinking3d, answer3d = await cot_agent_3d(cot_inputs_3d, subtask_3d_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3d.id}, refining cross-validation, thinking: {thinking3d.content}; answer: {answer3d.content}")
    sub_tasks.append(f"Sub-task 3d output: thinking - {thinking3d.content}; answer - {answer3d.content}")
    subtask_desc3d["response"] = {"thinking": thinking3d, "answer": answer3d}
    logs.append(subtask_desc3d)
    print("Step 3d: ", sub_tasks[-1])
    cot_instruction_4 = "Sub-task 4: Compare the validated exo product structure and stereochemistry from Subtask 3d with the provided multiple-choice options (A, B, C, D). Identify the choice that exactly matches the confirmed bicyclic core and stereochemical configuration, providing detailed justification for the selection."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3d", "answer of subtask 3d"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3d, answer3d], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, comparing validated product with choices, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    debate_instruction_5 = "Sub-task 5: Select and return the letter (A, B, C, or D) corresponding to the correct exo product from the provided choices based on the comparison and validation in Subtask 4."
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
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct exo product, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct exo product letter.", is_sub_task=True)
    agents.append(f"Final Decision agent on selecting exo product, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_instruction_6 = "Sub-task 6: Validate the final selected answer by cross-checking the chosen stereochemistry and structure against chemical databases or algorithmic stereochemistry tools to ensure no manual misassignments occurred."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, validating final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs