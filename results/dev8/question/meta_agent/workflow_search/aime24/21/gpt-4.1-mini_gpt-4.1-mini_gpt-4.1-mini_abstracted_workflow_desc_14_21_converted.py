async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1_1 = (
        "Sub-task 1: Identify and enumerate all lines in the regular dodecagon that can serve as sides of rectangles. "
        "Include all 12 polygon sides and all possible diagonals connecting non-adjacent vertices. "
        "Explicitly clarify and document assumptions: all diagonals are considered, rectangles must be fully inside the polygon, "
        "and degenerate or invalid lines are excluded. Verify uniqueness of each line and prepare a comprehensive list of these lines with identifiers (e.g., vertex indices). "
        "This subtask sets the foundation for all subsequent analysis by ensuring no relevant lines are omitted or misclassified."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, enumerating all polygon sides and diagonals, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_1_2 = (
        "Sub-task 2: Analyze the geometric properties of the identified lines: determine their directions (angles), lengths, and intersection points. "
        "Establish which lines are parallel or perpendicular based on the regular dodecagon's symmetry and uniform angular spacing. "
        "Compute all intersection points of diagonals inside the polygon, as these may serve as rectangle vertices. "
        "This step must produce explicit data structures representing line orientations and intersection coordinates to support algebraic and combinatorial rectangle detection."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1, answer_1_1],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, analyzing line properties and intersections, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 1: Derive formal coordinate representations of the polygon vertices, sides, and diagonals using coordinate geometry or complex numbers on the unit circle. "
        "Express all lines as vectors or linear equations to facilitate algebraic manipulation. Validate these representations by confirming known properties of the regular dodecagon (equal side lengths, vertex coordinates, symmetry). "
        "This formalization enables precise algebraic testing of rectangle conditions in the next steps."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_2, answer_1_2],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_2, answer_1_2], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, deriving formal coordinate representations, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_2_2 = (
        "Sub-task 2: Formulate explicit algebraic conditions for rectangle formation using the derived representations. "
        "Encode requirements that opposite sides are parallel and equal in length, and adjacent sides are perpendicular. "
        "Translate these geometric constraints into algebraic equations involving the identified lines and intersection points. "
        "This subtask must produce a clear, testable criterion for identifying rectangles formed by the polygon sides and diagonals."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking_2_1, answer_2_1],
        "agent_collaboration": "CoT"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2([taskInfo, thinking_2_1, answer_2_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, formulating rectangle conditions algebraically, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_3_1 = (
        "Sub-task 1: Systematically enumerate all rectangles formed inside the dodecagon by applying the rectangle conditions to all candidate quadruples of lines and intersection points. "
        "Explicitly distinguish between two classes of rectangles: (1) those formed strictly by polygon vertices (vertex-based rectangles), and (2) those formed by intersections of diagonals inside the polygon (intersection-based rectangles). "
        "For each rectangle, produce a concrete representation such as the coordinates of its four vertices or the indices of defining lines. "
        "Ensure no duplicates or degenerate rectangles are included. This enumeration must be exhaustive and rigorously documented to avoid assumptions or omissions."
    )
    N_sc = self.max_sc
    cot_sc_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking_2_2, answer_2_2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_3_1[i]([taskInfo, thinking_2_2, answer_2_2], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3_1[i].id}, enumerating rectangles with explicit representations, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3_1.append(answer_i)
        possible_thinkings_3_1.append(thinking_i)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + possible_answers_3_1 + possible_thinkings_3_1, "Sub-task 3.1: Synthesize and choose the most consistent and complete enumeration of rectangles.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 5: ", sub_tasks[-1])

    debate_instruction_3_2 = (
        "Sub-task 2: Verify the completeness and correctness of the enumerated rectangles. "
        "Perform an exhaustive search for any missing rectangles or degenerate cases by attempting to find counterexamples or additional rectangles not previously listed. "
        "Cross-validate the enumeration using symmetry arguments, combinatorial reasoning, or computational checks. "
        "Document the verification process and confirm that the final list of rectangles is complete and accurate."
    )
    reflexion_instruction_3_3 = (
        "Sub-task 3: Given the enumeration and verification, critically analyze the results to identify any overlooked errors, missing cases, or degenerate rectangles. "
        "Refine the enumeration if necessary and confirm the final list is robust and comprehensive."
    )
    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds = self.max_round
    all_thinking_3_2 = [[] for _ in range(N_rounds)]
    all_answer_3_2 = [[] for _ in range(N_rounds)]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instruction_3_2,
        "context": ["user query", thinking_3_1, answer_3_1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds):
        for i, agent in enumerate(debate_agents_3_2):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_3_1, answer_3_1], debate_instruction_3_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_3_1, answer_3_1] + all_thinking_3_2[r-1] + all_answer_3_2[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying enumeration completeness, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_3_2[r].append(thinking_i)
            all_answer_3_2[r].append(answer_i)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_2, answer_3_2 = await final_decision_agent_3_2([taskInfo] + all_thinking_3_2[-1] + all_answer_3_2[-1], "Sub-task 3.2: Finalize verification of rectangle enumeration.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 6: ", sub_tasks[-1])

    debate_instruction_4_1 = (
        "Sub-task 1: Count the total number of distinct rectangles identified in the verified enumeration. "
        "Decompose the count into meaningful components if applicable (e.g., vertex-based vs. intersection-based rectangles). "
        "Provide a clear, final answer with detailed justification referencing the explicit enumeration and verification results. "
        "This subtask consolidates all prior work into a definitive solution to the problem."
    )
    reflexion_instruction_4_2 = (
        "Sub-task 2: Perform a final critical review of the entire solution pipeline. "
        "Actively search for any overlooked errors, missing cases, or degenerate rectangles before accepting the final count. "
        "If discrepancies are found, return to earlier subtasks for correction. Otherwise, confirm the final answer as correct and robust. "
        "This step ensures the highest confidence in the solution's validity."
    )
    debate_agents_4_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_4 = self.max_round
    all_thinking_4_1 = [[] for _ in range(N_rounds_4)]
    all_answer_4_1 = [[] for _ in range(N_rounds_4)]
    subtask_desc_4_1 = {
        "subtask_id": "stage_4.subtask_1",
        "instruction": debate_instruction_4_1,
        "context": ["user query", thinking_3_2, answer_3_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_4):
        for i, agent in enumerate(debate_agents_4_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_3_2, answer_3_2], debate_instruction_4_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_3_2, answer_3_2] + all_thinking_4_1[r-1] + all_answer_4_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_4_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, counting and justifying final rectangle total, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_4_1[r].append(thinking_i)
            all_answer_4_1[r].append(answer_i)
    final_decision_agent_4_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4_1, answer_4_1 = await final_decision_agent_4_1([taskInfo] + all_thinking_4_1[-1] + all_answer_4_1[-1], "Sub-task 4.1: Finalize total rectangle count with justification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking_4_1.content}; answer - {answer_4_1.content}")
    subtask_desc_4_1['response'] = {"thinking": thinking_4_1, "answer": answer_4_1}
    logs.append(subtask_desc_4_1)
    print("Step 7: ", sub_tasks[-1])

    cot_reflect_instruction_4_2 = (
        "Sub-task 2: Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better by verifying and refining the final count of rectangles."
    )
    cot_agent_4_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4_2 = [taskInfo, thinking_4_1, answer_4_1]
    subtask_desc_4_2 = {
        "subtask_id": "stage_4.subtask_2",
        "instruction": cot_reflect_instruction_4_2,
        "context": ["user query", thinking_4_1, answer_4_1],
        "agent_collaboration": "Reflexion"
    }
    thinking_4_2, answer_4_2 = await cot_agent_4_2(cot_inputs_4_2, cot_reflect_instruction_4_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4_2.id}, critically reviewing final count, thinking: {thinking_4_2.content}; answer: {answer_4_2.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_4_2([taskInfo, thinking_4_2, answer_4_2], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4_2.extend([thinking_4_2, answer_4_2, feedback])
        thinking_4_2, answer_4_2 = await cot_agent_4_2(cot_inputs_4_2, cot_reflect_instruction_4_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4_2.id}, refining final count, thinking: {thinking_4_2.content}; answer: {answer_4_2.content}")
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking_4_2.content}; answer - {answer_4_2.content}")
    subtask_desc_4_2['response'] = {"thinking": thinking_4_2, "answer": answer_4_2}
    logs.append(subtask_desc_4_2)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_4_2, answer_4_2, sub_tasks, agents)
    return final_answer, logs
