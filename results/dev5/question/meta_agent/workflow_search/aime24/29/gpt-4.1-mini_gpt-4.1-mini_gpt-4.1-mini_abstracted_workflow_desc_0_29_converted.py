async def forward_29(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Clarify and formalize the problem constraints and assumptions with explicit attention to the maximality condition and the possibility of uncolored rows and columns. "
        "Specifically: interpret 'all chips in the same row and all chips in the same column have the same colour' as applying only to occupied cells; "
        "define maximality precisely as no empty cell can be filled without violating uniformity or chip availability; explicitly allow rows and columns to be assigned a 'none' color representing no chips placed; "
        "confirm that empty cells impose no color constraints; and establish that intersection cells are occupied only if the row and column colors agree. "
        "Avoid assuming all rows and columns must be colored. This subtask sets a precise foundation to prevent misinterpretation and overcounting."
    )
    subtask_id_1 = "subtask_1"
    print(f"Step 1: Starting {subtask_id_1}")
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": subtask_id_1,
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, clarifying problem constraints, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print(f"Step 1: {sub_tasks[-1]}")

    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze the structural implications of the refined constraints on the grid. "
        "Determine how the color assignments (black, white, or none) to rows and columns interact, especially at their intersections. "
        "Identify necessary and sufficient conditions for a cell to be occupied (row and column colors match and are not 'none'). "
        "Characterize how maximality requires that for each color, the set of colored rows is nonempty if and only if the set of colored columns is nonempty, ensuring coverage and preventing extendable placements. "
        "Understand how these conditions restrict possible placements and how uncolored rows/columns affect maximality. "
        "This includes reasoning about compatibility and coverage conditions to avoid invalid configurations. "
        "Use self-consistency by considering multiple reasoning paths to ensure robustness."
    )
    subtask_id_2 = "subtask_2"
    print(f"Step 2: Starting {subtask_id_2}")
    N_sc_2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": subtask_id_2,
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing structural constraints, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo] + possible_answers_2 + possible_thinkings_2,
        "Sub-task 2: Synthesize and choose the most consistent and correct analysis of structural constraints for the chip placement problem.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print(f"Step 2: {sub_tasks[-1]}")

    cot_instruction_3 = (
        "Sub-task 3: Enumerate all possible color assignments to the 5 rows and 5 columns, where each row and column can be assigned black, white, or none. "
        "For each assignment, determine the pattern of occupied cells (cells where row and column colors match and are not 'none'). "
        "Identify which assignments satisfy the uniformity and coverage conditions derived in subtask_2. "
        "This enumeration must consider the possibility of uncolored rows and columns and ensure that no colored row or column is left uncovered. "
        "The output is the set of all valid row-column color assignment pairs and their corresponding chip placements before considering chip quantity constraints."
    )
    subtask_id_3 = "subtask_3"
    print(f"Step 3: Starting {subtask_id_3}")
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": subtask_id_3,
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, enumerating valid assignments, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print(f"Step 3: {sub_tasks[-1]}")

    cot_instruction_4 = (
        "Sub-task 4: Incorporate chip quantity constraints into the analysis. "
        "For each valid assignment from subtask_3, calculate the number of black chips required (number of black rows × number of black columns) and white chips required (number of white rows × number of white columns). "
        "Filter out assignments that require more than 25 chips of either color, as these are infeasible given chip availability. "
        "This subtask ensures that only feasible configurations under chip quantity constraints are considered for maximality and counting."
    )
    subtask_id_4 = "subtask_4"
    print(f"Step 4: Starting {subtask_id_4}")
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": subtask_id_4,
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, filtering assignments by chip quantity, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print(f"Step 4: {sub_tasks[-1]}")

    cot_sc_instruction_5 = (
        "Sub-task 5: Formally characterize and verify the maximality condition for the filtered assignments from subtask_4. "
        "Confirm that no additional chip can be placed in any empty cell without violating uniformity or chip availability. "
        "Refine the set of valid assignments to include only those that are truly maximal. "
        "This involves checking that for each color, the coverage condition holds and that no empty cell can be added without conflict. "
        "This step finalizes the set of maximal configurations to be counted. "
        "Use self-consistency to verify robustness."
    )
    subtask_id_5 = "subtask_5"
    print(f"Step 5: Starting {subtask_id_5}")
    N_sc_5 = self.max_sc
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_5)]
    possible_answers_5 = []
    possible_thinkings_5 = []
    subtask_desc5 = {
        "subtask_id": subtask_id_5,
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_5):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking4, answer4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, verifying maximality, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5)
        possible_thinkings_5.append(thinking5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo] + possible_answers_5 + possible_thinkings_5,
        "Sub-task 5: Synthesize and select the most consistent characterization of maximal configurations.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print(f"Step 5: {sub_tasks[-1]}")

    cot_instruction_6 = (
        "Sub-task 6: Count the number of valid maximal configurations obtained from subtask_5. "
        "Aggregate the counts considering indistinguishability of chips within colors and any symmetries or equivalences in assignments. "
        "Provide the final numeric answer representing the number of ways to place chips on the grid under all given conditions, including maximality and chip availability."
    )
    subtask_id_6 = "subtask_6"
    print(f"Step 6: Starting {subtask_id_6}")
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": subtask_id_6,
        "instruction": cot_instruction_6,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, counting valid maximal configurations, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print(f"Step 6: {sub_tasks[-1]}")

    cot_reflect_instruction_7 = (
        "Sub-task 7: Verify the correctness of the entire reasoning process and the final count. "
        "Cross-check assumptions, constraints, and calculations for consistency with the problem statement and prior subtasks. "
        "Introduce a reflexion step where agents explicitly challenge and debate the maximality definition, coverage conditions, and chip quantity constraints to detect any remaining oversights. "
        "If discrepancies or ambiguities are found, refine previous subtasks accordingly and update the final count. "
        "Provide a final verified answer with justification."
    )
    subtask_id_7 = "subtask_7"
    print(f"Step 7: Starting {subtask_id_7}")
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6, answer6]
    subtask_desc7 = {
        "subtask_id": subtask_id_7,
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", thinking6.content, answer6.content],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, verifying final count, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7],
                                               "Please review and provide limitations of the provided solution. If absolutely correct, output exactly 'True' in 'correct'.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining final count, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print(f"Step 7: {sub_tasks[-1]}")

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
