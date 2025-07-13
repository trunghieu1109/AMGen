async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Establish a concrete coordinate system and vector representation for the convex equilateral hexagon ABCDEF with opposite sides parallel. "
        "Place vertex A at the origin (0,0). Define the hexagon side length as s (unknown) and introduce an angle parameter theta to represent the direction of side AB and adjacent sides. "
        "Define vectors for sides AB, BC, and CD as v1 = (s, 0), v2 = (s*cos(theta), s*sin(theta)), and v3 = -v1 - v2, ensuring the hexagon closes and opposite sides are parallel. "
        "Compute coordinates of all vertices B, C, D, E, and F accordingly. Explicitly state all assumptions and avoid assuming specific numeric values for theta or s at this stage. "
        "This setup will facilitate symbolic algebraic manipulation in subsequent subtasks."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, establishing coordinate system and vector representation, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Derive parametric equations of the lines containing sides AB, CD, and EF using the vertex coordinates from subtask_1. "
        "Express these lines in vector-parametric form suitable for symbolic intersection computations. Avoid assuming any numeric values for parameters. "
        "Confirm that these lines are well-defined and correspond to the hexagon sides extended infinitely. This subtask prepares the groundwork for explicit intersection point calculations."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, deriving parametric line equations, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_1_1 = (
        "Sub-task 1: Symbolically compute the intersection points P = AB ∩ CD, Q = CD ∩ EF, and R = EF ∩ AB by solving the parametric line equations derived in stage_0.subtask_2. "
        "Express each intersection point explicitly as functions of s and theta. Ensure algebraic correctness and provide explicit formulae for the coordinates of P, Q, and R. "
        "Avoid heuristic or approximate methods; all derivations must be symbolic and exact."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, computing intersection points P, Q, R, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_1_2 = (
        "Sub-task 2: Derive symbolic expressions for the squared distances |P-Q|^2, |Q-R|^2, and |R-P|^2 in terms of s and theta using the coordinates of P, Q, and R from subtask_1. "
        "Simplify these expressions as much as possible to facilitate solving. Emphasize algebraic rigor and avoid skipping steps or relying on numeric approximations. "
        "These expressions represent the squared side lengths of the triangle formed by the extended lines AB, CD, and EF."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, deriving squared distance expressions, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_1_3 = (
        "Sub-task 3: Formulate the system of equations by equating the derived squared distances to the squares of the given triangle side lengths: |P-Q| = 200, |Q-R| = 240, and |R-P| = 300. "
        "Explicitly write the system as equations in s and theta. Prepare the system for algebraic or numeric solving, ensuring all variables and parameters are clearly defined. "
        "Avoid introducing extraneous variables or assumptions. This subtask sets up the core algebraic problem to find s and theta."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_3 = {
        "subtask_id": "subtask_3",
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
        "Sub-task 1: Solve the system of equations from stage_1.subtask_3 algebraically and/or numerically to find all physically valid solutions for s (the hexagon side length) and theta (the angle parameter). "
        "Use appropriate algebraic manipulation, substitution, or numeric root-finding methods. Explicitly document the solving process, including any assumptions or constraints applied to discard non-physical or extraneous solutions (e.g., negative lengths, invalid angles). "
        "Provide numeric approximations with sufficient precision and symbolic expressions where possible."
    )
    N_sc = self.max_sc
    cot_sc_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_1, answer_2_1 = await cot_sc_agents_2_1[i]([taskInfo, thinking_1_3, answer_1_3], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_1[i].id}, solving system for s and theta, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
        possible_answers_2_1.append(answer_2_1)
        possible_thinkings_2_1.append(thinking_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_answers_2_1 + possible_thinkings_2_1, "Sub-task 1: Synthesize and choose the most consistent and correct solutions for s and theta.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing solutions for s and theta, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 6: ", sub_tasks[-1])

    cot_reflect_instruction_3_1 = (
        "Sub-task 1: Verify the candidate solutions for s and theta by back-substituting into the distance formulas and checking that the computed triangle side lengths match 200, 240, and 300 within acceptable numeric tolerance. "
        "Additionally, confirm that the hexagon constructed with these parameters is convex, equilateral, and has opposite sides parallel as required. Reject any solutions failing these geometric constraints. "
        "Provide a final, justified answer for the hexagon side length s, including numeric value and verification details. Avoid accepting heuristic or approximate answers without rigorous validation. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round
    cot_inputs_3_1 = [taskInfo, thinking_0_1, answer_0_1, thinking_2_1, answer_2_1]
    subtask_desc_3_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_reflect_instruction_3_1,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1(cot_inputs_3_1, cot_reflect_instruction_3_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, verifying and refining solutions, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    for i in range(N_max_reflect):
        feedback_3_1, correct_3_1 = await critic_agent_3_1([taskInfo, thinking_3_1, answer_3_1], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_1.id}, providing feedback, thinking: {feedback_3_1.content}; answer: {correct_3_1.content}")
        if correct_3_1.content == "True":
            break
        cot_inputs_3_1.extend([thinking_3_1, answer_3_1, feedback_3_1])
        thinking_3_1, answer_3_1 = await cot_agent_3_1(cot_inputs_3_1, cot_reflect_instruction_3_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, refining solutions, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
