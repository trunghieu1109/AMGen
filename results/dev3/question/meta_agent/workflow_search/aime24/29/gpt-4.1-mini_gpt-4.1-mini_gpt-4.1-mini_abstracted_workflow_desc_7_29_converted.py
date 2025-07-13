async def forward_29(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_reflexion_instruction_1 = (
        "Sub-task 1: Formally represent the problem setup and constraints for the chip placement problem. "
        "Define variables for row and column color assignments as functions from {1,...,5} to {white, black, empty}. "
        "Explicitly express conditions: (1) each cell contains at most one chip, (2) all chips in the same row have the same color, "
        "(3) all chips in the same column have the same color, and (4) the color at the intersection cell must be consistent if occupied. "
        "Crucially, formalize the maximality condition precisely: for each color assigned to any row, there must be at least one column assigned the same color, and vice versa, forbidding isolated colors without matching counterparts. "
        "Clarify assumptions about empty rows or columns and the interpretation of maximality, emphasizing that no additional chip can be added without violating uniformity. "
        "Avoid assuming any color assignments without justification. This formalization will serve as the foundation for all subsequent reasoning and enumeration."
    )
    cot_reflexion_agent_1 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_reflexion_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "Reflexion"
    }
    thinking1, answer1 = await cot_reflexion_agent_1([taskInfo], cot_reflexion_instruction_1, 0, is_sub_task=True)
    agents.append(f"Reflexion agent {cot_reflexion_agent_1.id}, formal representation and maximality formalization, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    cot_sc_instruction_2 = (
        "Sub-task 2: Validate the formal representations from Sub-task 1 by checking internal consistency and logical implications. "
        "Confirm that the color uniformity constraints imply the grid's chip placement corresponds to a matrix where each occupied cell's color equals both its row and column color. "
        "Analyze how maximality restricts empty cells and color assignments. Explicitly identify and enumerate the four invalid maximality patterns that must be excluded: "
        "(a) rows assigned white but no columns assigned white, "
        "(b) columns assigned white but no rows assigned white, "
        "(c) rows assigned black but no columns assigned black, "
        "(d) columns assigned black but no rows assigned black. "
        "This step ensures the model accurately captures the problem and is ready for rigorous enumeration."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, validate formal representation and enumerate invalid maximality patterns, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    best_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    cot_sc_instruction_3 = (
        "Sub-task 3: Characterize all possible assignments of colors to rows and columns under the constraints derived in Stage 1, explicitly enforcing maximality. "
        "Enumerate all functions from rows to {white, black, empty} and columns to {white, black, empty} such that: "
        "(1) the intersection cells' colors are consistent, "
        "(2) the maximality condition is strictly enforced by excluding the four invalid patterns identified in Subtask 2, and "
        "(3) no additional chip can be added without violating uniformity. "
        "Provide a detailed combinatorial description of valid row and column color patterns, including symbolic variables representing counts of rows and columns assigned each color. "
        "Avoid double counting configurations equivalent due to indistinguishability of chips. This characterization will form the basis for precise counting."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, characterize valid row and column color assignments enforcing maximality, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    best_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[best_answer_3]
    answer3 = answermapping_3[best_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    debate_instruction_4 = (
        "Sub-task 4: Verify the characterization from Sub-task 3 for compliance with all problem constraints, including maximality and color consistency at intersections. "
        "Employ a structured debate among agents to explicitly propose, cross-check, and exclude any assignments violating maximality, ensuring all four invalid maximality patterns and their symmetric cases are accounted for. "
        "Confirm that the characterization is complete and no valid configurations are omitted. This verification step prevents propagation of errors and ensures only valid configurations proceed to counting."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying maximality enforcement and excluding invalid patterns, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Final verification of valid assignments ensuring maximality and consistency.", is_sub_task=True)
    agents.append(f"Final Decision agent, final verification of valid assignments, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)

    reflexion_instruction_5 = (
        "Sub-task 5: Combine the verified row and column color assignments from Sub-task 4 to count the total number of distinct valid chip placements on the grid. "
        "Derive a closed-form formula or explicit count based on the combinatorial structure identified, rigorously applying inclusion–exclusion to account for all four maximality violation events and their intersections. "
        "Pass symbolic variables (e.g., counts of rows/columns with/without each color) explicitly in the reasoning. "
        "Ensure the counting respects indistinguishability of chips and maximality constraints. Avoid partial or incomplete inclusion–exclusion corrections that led to overcounting in previous attempts."
    )
    cot_reflect_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": reflexion_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_reflect_agent_5(cot_inputs_5, reflexion_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect_agent_5.id}, combine and count configurations with inclusion-exclusion, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5],
                                               "Please review the answer above and criticize any errors or omissions. If correct, output exactly 'True' in 'correct'.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_reflect_agent_5(cot_inputs_5, reflexion_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect_agent_5.id}, refined count, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)

    reflexion_instruction_6 = (
        "Sub-task 6: Perform a final verification and synthesis of the counting result from Sub-task 5. "
        "Cross-check the derived count against problem constraints and logical expectations, including automated validation on smaller grids (e.g., 2x2 or 3x3) to confirm correctness of the inclusion-exclusion formula and maximality enforcement. "
        "Engage agents in reflexion and debate to challenge assumptions, verify maximality enforcement, and validate counting correctness. "
        "Provide the final answer with detailed justification, confirming that no valid configurations have been omitted or double counted. "
        "This step ensures the solution's correctness, completeness, and robustness."
    )
    cot_reflect_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": reflexion_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_reflect_agent_6(cot_inputs_6, reflexion_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect_agent_6.id}, final verification and synthesis, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6],
                                               "Please review the final answer and provide feedback. If correct, output exactly 'True' in 'correct'.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_reflect_agent_6(cot_inputs_6, reflexion_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect_agent_6.id}, refined final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    for i, st in enumerate(sub_tasks, 1):
        print(f"Step {i}: ", st)
    return final_answer, logs
