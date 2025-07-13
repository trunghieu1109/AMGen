async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0_1 = (
        "Sub-task 1: Identify and rigorously verify the geometric properties and constraints of the hexagon ABCDEF. "
        "Confirm that ABCDEF is a convex equilateral hexagon with all pairs of opposite sides parallel (AB || DE, BC || EF, CD || FA). "
        "Enumerate the implications of these properties on the shape and side directions. Clarify the labeling and orientation of vertices and sides, "
        "ensuring no assumptions about coordinate systems or angles are made at this stage. Understand the nature and formation of the triangle created by the extensions of sides AB, CD, and EF, "
        "including the fact that these three lines intersect pairwise to form a triangle with side lengths 200, 240, and 300. Avoid assuming the triangle's position relative to the hexagon or any numeric values beyond those given."
    )
    N_sc = self.max_sc
    cot_sc_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_sc_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_1[i].id}, verifying geometric elements, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)

    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1(
        [taskInfo] + possible_answers_0_1 + possible_thinkings_0_1,
        "Sub-task 1: Synthesize and choose the most consistent and correct verification of geometric elements and constraints.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Analyze the relationship between the triangle formed by the extended lines AB, CD, and EF and the hexagon. "
        "Confirm that the triangle's side lengths correspond to distances between the intersection points of these extended lines. "
        "Establish that these lines intersect pairwise to form a triangle and clarify the geometric meaning of these intersection points. "
        "Avoid assuming the triangle lies inside or outside the hexagon without explicit justification. Prepare the groundwork for symbolic representation by summarizing all known constraints and relationships."
    )
    cot_sc_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_sc_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_2[i].id}, verifying triangle-hexagon relation, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_2.append(answer)
        possible_thinkings_0_2.append(thinking)

    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2(
        [taskInfo] + possible_answers_0_2 + possible_thinkings_0_2,
        "Sub-task 2: Synthesize and choose the most consistent and correct verification of triangle and hexagon relationship.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_1_1 = (
        "Sub-task 1: Define a concrete 2D coordinate system and derive symbolic vector representations for the hexagon sides and their directions, "
        "strictly respecting the problem's constraints. Assign vectors u, v, w to represent directions of sides AB, BC, and CD respectively, ensuring that opposite sides are parallel and all sides have equal length s. "
        "Avoid assuming fixed angles such as 120 degrees between vectors unless rigorously justified. Express all hexagon vertices in terms of these vectors and s, maintaining symbolic form. "
        "This step must produce a consistent, general vector model of the hexagon suitable for further algebraic manipulation."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, deriving vector representations, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_1_2 = (
        "Sub-task 2: Using the vector representations from subtask_1, derive explicit parametric equations for the lines extending sides AB, CD, and EF. "
        "Symbolically compute the exact intersection points of these extended lines by solving the parametric equations rigorously, expressing each intersection point as a function of s and the direction vectors. "
        "Avoid numerical approximations or heuristic assumptions. Derive symbolic expressions for the distances between these intersection points, which correspond to the triangle sides, in terms of s and vector parameters."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1, thinking_0_2, answer_0_2], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, deriving line intersections, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_1_3 = (
        "Sub-task 3: Formulate the system of equations relating the symbolic expressions of the triangle side lengths (from subtask_2) to the given numerical values 200, 240, and 300. "
        "Prepare these equations for numeric solving, ensuring all variables and parameters are clearly defined. Avoid premature numeric solving or ignoring geometric constraints. "
        "This subtask bridges symbolic derivation and numeric solution stages."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_1_2, answer_1_2], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, formulating system of equations, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Numerically solve the system of equations derived in stage_1.subtask_3 to find candidate values for the hexagon side length s. "
        "Implement numeric methods with sufficient precision and rigor. For each candidate solution, perform intermediate numeric verification steps to check consistency with all geometric constraints, including convexity of the hexagon, parallelism of opposite sides, and positivity of side length. "
        "Discard any invalid or extraneous solutions. Document the numeric verification process and results."
    )
    cot_sc_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_sc_agents_2_1[i]([taskInfo, thinking_1_3, answer_1_3], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_1[i].id}, solving equations for hexagon side length, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2_1.append(answer)
        possible_thinkings_2_1.append(thinking)

    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1(
        [taskInfo] + possible_answers_2_1 + possible_thinkings_2_1,
        "Sub-task 1: Synthesize and choose the most consistent and correct hexagon side length from equations.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_2_2 = (
        "Sub-task 2: Verify the geometric plausibility of the candidate solutions from subtask_1 by checking that the three intersection points form a valid triangle consistent with the problem statement. "
        "Confirm that the triangle lies in a position compatible with the convex equilateral hexagon and that the solution respects all problem constraints. "
        "This includes verifying that the triangle side lengths match the given values within acceptable numeric tolerance and that the hexagon remains convex with parallel opposite sides. "
        "Prepare a summary of verification results to support final solution acceptance."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2([taskInfo, thinking_2_1, answer_2_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, verifying geometric plausibility, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 7: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = (
        "Sub-task 1: Finalize the solution by simplifying the computed hexagon side length s to its minimal exact or numeric form. "
        "Provide the final numeric value with appropriate precision. Reflect on the solution's geometric correctness and consistency with all problem constraints, including equilateral sides, parallelism, convexity, and the triangle side lengths. "
        "Present a comprehensive verification report confirming the solution's validity. Avoid leaving the answer ambiguous or unsupported by verification. "
        + reflect_inst
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_1 = self.max_round
    cot_inputs = [taskInfo, thinking_2_2, answer_2_2]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_reflect_instruction,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, finalizing and verifying solution, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    for i in range(N_max_3_1):
        feedback, correct = await critic_agent_3_1([taskInfo, thinking_3_1, answer_3_1],
                                                  "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_1.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking_3_1, answer_3_1, feedback])
        thinking_3_1, answer_3_1 = await cot_agent_3_1(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, refining solution, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
