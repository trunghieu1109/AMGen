async def forward_50(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Understand the physical setup: a point charge q placed at a distance d from the center of a grounded conducting sphere of radius R. Identify the relevant electrostatic principles, specifically the method of image charges, applicable to this configuration."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding physical setup, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Determine the magnitude and position of the image charge inside the grounded conducting sphere that replicates the boundary conditions on the sphere's surface for the external charge q located at distance d, based on the physical setup from Sub-task 1."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining image charge magnitude and position, thinking: {thinking2.content}; answer: {answer2.content}")
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
    subtask_desc3_1 = {
        "subtask_id": "subtask_3_1",
        "instruction": "Sub-task 3.1: Calculate explicitly the distance between the real charge q (at distance d) and the image charge (at its determined position inside the sphere).",
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await cot_agent_3_1([taskInfo, thinking2, answer2], subtask_desc3_1["instruction"], is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, calculating distance between charges, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {
        "thinking": thinking3_1,
        "answer": answer3_1
    }
    logs.append(subtask_desc3_1)
    print("Step 3.1: ", sub_tasks[-1])
    subtask_desc3_2 = {
        "subtask_id": "subtask_3_2",
        "instruction": "Sub-task 3.2: Write down the formula for the electrostatic interaction energy between the real charge and the image charge, clearly defining all variables and constants.",
        "context": ["user query", "thinking of subtask 3_1", "answer of subtask 3_1"],
        "agent_collaboration": "CoT"
    }
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3_2, answer3_2 = await cot_agent_3_2([taskInfo, thinking3_1, answer3_1], subtask_desc3_2["instruction"], is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_2.id}, writing interaction energy formula, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc3_2['response'] = {
        "thinking": thinking3_2,
        "answer": answer3_2
    }
    logs.append(subtask_desc3_2)
    print("Step 3.2: ", sub_tasks[-1])
    subtask_desc3_3 = {
        "subtask_id": "subtask_3_3",
        "instruction": "Sub-task 3.3: Substitute the expressions for the image charge magnitude, its position, and the distance between charges into the interaction energy formula. Simplify the algebraic expression step-by-step, carefully tracking powers of R and d.",
        "context": ["user query", "thinking of subtask 3_2", "answer of subtask 3_2", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3_1", "answer of subtask 3_1"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_3_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_3 = self.max_round
    cot_inputs_3_3 = [taskInfo, thinking2, answer2, thinking3_1, answer3_1, thinking3_2, answer3_2]
    thinking3_3, answer3_3 = await cot_agent_3_3(cot_inputs_3_3, subtask_desc3_3["instruction"], 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_3.id}, substituting and simplifying interaction energy, thinking: {thinking3_3.content}; answer: {answer3_3.content}")
    for i in range(N_max_3_3):
        feedback, correct = await critic_agent_3_3([taskInfo, thinking3_3, answer3_3], "Please review the algebraic simplification and correctness of the interaction energy expression, focusing on powers of R and d and the factor 1/2.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_3.extend([thinking3_3, answer3_3, feedback])
        thinking3_3, answer3_3 = await cot_agent_3_3(cot_inputs_3_3, subtask_desc3_3["instruction"], i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_3.id}, refining interaction energy expression, thinking: {thinking3_3.content}; answer: {answer3_3.content}")
    sub_tasks.append(f"Sub-task 3.3 output: thinking - {thinking3_3.content}; answer - {answer3_3.content}")
    subtask_desc3_3['response'] = {
        "thinking": thinking3_3,
        "answer": answer3_3
    }
    logs.append(subtask_desc3_3)
    print("Step 3.3: ", sub_tasks[-1])
    subtask_desc3_4 = {
        "subtask_id": "subtask_3_4",
        "instruction": "Sub-task 3.4: Apply the factor of 1/2 to the interaction energy to account for the grounded conducting sphere boundary condition, explaining the physical justification for this factor.",
        "context": ["user query", "thinking of subtask 3_3", "answer of subtask 3_3"],
        "agent_collaboration": "CoT"
    }
    cot_agent_3_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3_4, answer3_4 = await cot_agent_3_4([taskInfo, thinking3_3, answer3_3], subtask_desc3_4["instruction"], is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_4.id}, applying factor 1/2 and explaining physical justification, thinking: {thinking3_4.content}; answer: {answer3_4.content}")
    sub_tasks.append(f"Sub-task 3.4 output: thinking - {thinking3_4.content}; answer - {answer3_4.content}")
    subtask_desc3_4['response'] = {
        "thinking": thinking3_4,
        "answer": answer3_4
    }
    logs.append(subtask_desc3_4)
    print("Step 3.4: ", sub_tasks[-1])
    subtask_desc3_5 = {
        "subtask_id": "subtask_3_5",
        "instruction": "Sub-task 3.5: Perform a self-consistency check by independently deriving the net potential energy expression through at least two different approaches or agents, then compare and reconcile any discrepancies before finalizing the expression.",
        "context": ["user query", "thinking of subtask 3_4", "answer of subtask 3_4"],
        "agent_collaboration": "SC_CoT"
    }
    N_sc_3_5 = self.max_sc
    cot_agents_3_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3_5)]
    possible_answers_3_5 = []
    thinkingmapping_3_5 = {}
    answermapping_3_5 = {}
    for i in range(N_sc_3_5):
        thinking3_5, answer3_5 = await cot_agents_3_5[i]([taskInfo, thinking3_4, answer3_4], subtask_desc3_5["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_5[i].id}, independently deriving net potential energy, thinking: {thinking3_5.content}; answer: {answer3_5.content}")
        possible_answers_3_5.append(answer3_5.content)
        thinkingmapping_3_5[answer3_5.content] = thinking3_5
        answermapping_3_5[answer3_5.content] = answer3_5
    answer3_5_content = Counter(possible_answers_3_5).most_common(1)[0][0]
    thinking3_5 = thinkingmapping_3_5[answer3_5_content]
    answer3_5 = answermapping_3_5[answer3_5_content]
    sub_tasks.append(f"Sub-task 3.5 output: thinking - {thinking3_5.content}; answer - {answer3_5.content}")
    subtask_desc3_5['response'] = {
        "thinking": thinking3_5,
        "answer": answer3_5
    }
    logs.append(subtask_desc3_5)
    print("Step 3.5: ", sub_tasks[-1])
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": "Sub-task 4: Compare the finalized derived expression for the net potential energy term-by-term (numerator and denominator) with each of the provided multiple-choice options. Focus on the algebraic form, especially the powers and placement of R and d, the presence of the factor 1/2, and dimensional consistency.",
        "context": ["user query", "thinking of subtask 3_5", "answer of subtask 3_5"],
        "agent_collaboration": "CoT"
    }
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3_5, answer3_5], subtask_desc4["instruction"], is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, comparing derived expression with options, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    debate_instruction_5 = "Sub-task 5: Select the correct multiple-choice answer (A, B, C, or D) based on the formal algebraic matching and physical consistency of the derived net potential energy expression with the given options. If any agent flags a mismatch, trigger a re-verification loop."
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
        minority_flag = False
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct choice, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
            if answer5.content not in ['A', 'B', 'C', 'D']:
                minority_flag = True
        if not minority_flag:
            break
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct multiple-choice answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
