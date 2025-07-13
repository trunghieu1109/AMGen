async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Geometric Setup and Parametrization

    # Sub-task 1: Define coordinate system and vector u for AB
    cot_instruction_0_1 = (
        "Sub-task 1: Define a coordinate system by placing point A of the hexagon ABCDEF at the origin. "
        "Introduce a vector u representing the direction of side AB with an unknown angle θ relative to the x-axis. "
        "Establish vector notation for the hexagon sides, starting with AB = s * u, where s is the unknown side length. "
        "Avoid assuming any specific numeric values or angles at this stage."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage0_subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, defining coordinate system and vector u, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    # Sub-task 2: Express directions of other sides ensuring parallelism
    cot_instruction_0_2 = (
        "Sub-task 2: Express the directions of the other sides of the hexagon (BC, CD, DE, EF, FA) in terms of vectors u and v, "
        "where v is another vector defined relative to u and θ, ensuring that opposite sides are parallel (i.e., AB || DE, BC || EF, CD || FA). "
        "Formulate these parallelism constraints explicitly using vector relations without assuming regularity of angles."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage0_subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, expressing directions and parallelism, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    # Sub-task 3: Formulate equilateral and convexity conditions
    cot_instruction_0_3 = (
        "Sub-task 3: Formulate the equilateral condition by setting the magnitudes of all side vectors equal to the unknown side length s. "
        "Additionally, state the convexity condition of the hexagon in terms of the vectors and angles, ensuring the polygon is convex without assuming regularity. "
        "Avoid mixing these conditions with the triangle formed by extended sides at this stage."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage0_subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_2, answer_0_2], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, formulating equilateral and convexity conditions, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)

    # Sub-task 4: Derive parametric equations for lines AB, CD, EF
    cot_instruction_0_4 = (
        "Sub-task 4: Derive parametric equations for the lines containing sides AB, CD, and EF. "
        "Use their base points (A, C, E) and direction vectors (u, w, -v) as established in previous subtasks. "
        "Express each line in parametric form L(t) = P + t * d, where P is a base point and d is a direction vector. "
        "Avoid premature assumptions about intersection points or triangle side lengths."
    )
    cot_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_4 = {
        "subtask_id": "stage0_subtask_4",
        "instruction": cot_instruction_0_4,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_4, answer_0_4 = await cot_agent_0_4([taskInfo, thinking_0_2, answer_0_2], cot_instruction_0_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_4.id}, deriving parametric line equations, thinking: {thinking_0_4.content}; answer: {answer_0_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)

    # Sub-task 5: Compute intersection points P, Q, R of extended lines
    cot_instruction_0_5 = (
        "Sub-task 5: Compute the intersection points P, Q, and R of the pairs of extended lines (AB and CD), (CD and EF), and (EF and AB) respectively by solving their parametric equations symbolically in terms of s and θ. "
        "Provide explicit expressions for these points without numeric substitution. Ensure the solution accounts for the correct ordering and existence of intersections forming a triangle."
    )
    cot_agent_0_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_5 = {
        "subtask_id": "stage0_subtask_5",
        "instruction": cot_instruction_0_5,
        "context": ["user query", thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_5, answer_0_5 = await cot_agent_0_5([taskInfo, thinking_0_4, answer_0_4], cot_instruction_0_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_5.id}, computing intersection points, thinking: {thinking_0_5.content}; answer: {answer_0_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_0_5.content}; answer - {answer_0_5.content}")
    subtask_desc_0_5['response'] = {"thinking": thinking_0_5, "answer": answer_0_5}
    logs.append(subtask_desc_0_5)

    # Sub-task 6: Express triangle side vectors and lengths from intersection points
    cot_instruction_0_6 = (
        "Sub-task 6: Express the side vectors of the triangle formed by points P, Q, and R as differences of their coordinates (e.g., vector PQ = Q - P). "
        "Formulate the side lengths of this triangle as Euclidean distances involving s and θ. "
        "Avoid simplifying these expressions prematurely or assuming direct proportionality to hexagon side vectors."
    )
    cot_agent_0_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_6 = {
        "subtask_id": "stage0_subtask_6",
        "instruction": cot_instruction_0_6,
        "context": ["user query", thinking_0_5.content, answer_0_5.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_6, answer_0_6 = await cot_agent_0_6([taskInfo, thinking_0_5, answer_0_5], cot_instruction_0_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_6.id}, expressing triangle side vectors and lengths, thinking: {thinking_0_6.content}; answer: {answer_0_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_0_6.content}; answer - {answer_0_6.content}")
    subtask_desc_0_6['response'] = {"thinking": thinking_0_6, "answer": answer_0_6}
    logs.append(subtask_desc_0_6)

    # Stage 1: Algebraic System Setup and Simplification

    # Sub-task 1: Set up system equating triangle side lengths to given lengths
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Set up the system of equations by equating the triangle side lengths derived from the intersection points (expressed in terms of s and θ) to the given lengths 200, 240, and 300. "
        "Ensure the system correctly reflects the geometric constraints without oversimplification. Avoid mixing these equations with hexagon side length equalities at this stage."
    )
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage1_subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_6.content, answer_0_6.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0_6, answer_0_6], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, setting up system of equations, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent system of equations for the triangle side lengths.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing system of equations, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    # Sub-task 2: Combine equilateral and parallelism constraints with triangle equations
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Combine the equilateral and parallelism constraints of the hexagon (from stage_0_subtask_3) with the triangle side length equations (from subtask_1) to form a complete algebraic system in variables s and θ. "
        "Simplify the system using vector operations and trigonometric identities as needed, ensuring no loss of geometric meaning or validity."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage1_subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0_3.content, answer_0_3.content, thinking_1_1, answer_1_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_0_3, answer_0_3, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, combining constraints and simplifying system, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent combined algebraic system.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing combined algebraic system, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # Sub-task 3: Analyze solvability and reduce system
    cot_reflect_instruction_1_3 = (
        "Sub-task 3: Analyze the solvability and consistency of the algebraic system. "
        "If necessary, apply substitution or elimination methods to reduce the system to a solvable form for s and θ. "
        "Document any assumptions or constraints required to ensure a unique and valid solution corresponding to a convex hexagon and a valid triangle. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking_0_3, answer_0_3, thinking_1_2, answer_1_2]
    subtask_desc_1_3 = {
        "subtask_id": "stage1_subtask_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking_0_3.content, answer_0_3.content, thinking_1_2, answer_1_2],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, analyzing solvability, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    critic_inst_1_3 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_1_3):
        feedback_1_3, correct_1_3 = await critic_agent_1_3([taskInfo, thinking_1_3, answer_1_3], critic_inst_1_3, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, providing feedback, thinking: {feedback_1_3.content}; answer: {correct_1_3.content}")
        if correct_1_3.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, answer_1_3, feedback_1_3])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining solvability analysis, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    # Stage 2: Solve and Verify

    # Sub-task 1: Solve algebraic system for s and θ
    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Solve the simplified algebraic system to find the numerical values of the hexagon's side length s and the angle θ. "
        "Verify that the solution satisfies all geometric constraints, including convexity, parallelism, and equilateral conditions of the hexagon, as well as the given triangle side lengths. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking_1_3, answer_1_3]
    subtask_desc_2_1 = {
        "subtask_id": "stage2_subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, solving algebraic system, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    critic_inst_2_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_2_1):
        feedback_2_1, correct_2_1 = await critic_agent_2_1([taskInfo, thinking_2_1, answer_2_1], critic_inst_2_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback_2_1.content}; answer: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, answer_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining solution, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    # Sub-task 2: Verify solution by reconstructing hexagon and triangle
    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Perform a comprehensive verification of the solution by reconstructing the hexagon and the triangle using the found values of s and θ. "
        "Check the convexity of the hexagon, the parallelism of opposite sides, the equality of side lengths, and the accuracy of the triangle side lengths formed by the extended lines. "
        "Confirm that the solution is geometrically valid and consistent. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage2_subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, verifying solution, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    critic_inst_2_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], critic_inst_2_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback_2_2.content}; answer: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining verification, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
