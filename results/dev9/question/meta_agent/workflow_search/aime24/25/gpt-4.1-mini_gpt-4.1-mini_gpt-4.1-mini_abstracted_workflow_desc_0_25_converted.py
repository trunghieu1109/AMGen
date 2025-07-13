async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Identify and verify the key geometric elements and constraints of the problem. "
        "Confirm that ABCDEF is a convex equilateral hexagon with three pairs of opposite sides parallel (AB ∥ DE, BC ∥ EF, CD ∥ FA). "
        "Define the side length s as the common length of all sides. Clarify the nature of the triangle formed by the extensions of sides AB, CD, and EF, "
        "interpreting the given side lengths 200, 240, and 300 as distances between intersection points of these extended lines. "
        "Avoid assumptions beyond the given convexity and parallelism conditions. This subtask sets the foundational understanding and notation for all subsequent steps."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing geometric elements, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Assign a fixed coordinate system to the hexagon to enable explicit algebraic and numeric computations. "
        "Place point A at the origin (0,0) and point B at (s,0), where s is the unknown side length. "
        "Express points C, D, E, and F in terms of s and unknown angles or vectors consistent with the hexagon's properties (equilateral, convex, and with opposite sides parallel). "
        "Output explicit coordinate expressions for all vertices in symbolic form, clearly defining all variables and parameters. "
        "This step eliminates ambiguity in vector directions and sets the stage for precise line and intersection calculations."
    )
    N_sc = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, assigning coordinates, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo] + possible_answers_1_1 + possible_thinkings_1_1,
        "Sub-task 1: Synthesize and choose the most consistent coordinate assignment for the hexagon.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Derive explicit parametric or general equations for the lines containing sides AB, CD, and EF using the coordinates assigned in subtask_1. "
        "Express these line equations symbolically in terms of s and any angle parameters introduced. Ensure the equations are suitable for symbolic intersection computations. "
        "This step bridges vertex coordinates to line representations necessary for intersection and distance calculations."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, deriving line equations, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo] + possible_answers_1_2 + possible_thinkings_1_2,
        "Sub-task 2: Synthesize and choose the most consistent line equations for AB, CD, and EF.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_1_3 = (
        "Sub-task 3: Compute symbolically the intersection points of the pairs of extended lines: P1 = AB ∩ CD, P2 = CD ∩ EF, and P3 = EF ∩ AB. "
        "Express the coordinates of P1, P2, and P3 explicitly in terms of s and the angle parameters. Provide detailed algebraic derivations and intermediate expressions. "
        "This step concretely links the hexagon’s geometry to the triangle formed by these intersections."
    )
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_3, answer_1_3 = await cot_agents_1_3[i]([taskInfo, thinking_1_2, answer_1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, computing intersections, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
        possible_answers_1_3.append(answer_1_3)
        possible_thinkings_1_3.append(thinking_1_3)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3(
        [taskInfo] + possible_answers_1_3 + possible_thinkings_1_3,
        "Sub-task 3: Synthesize and choose the most consistent intersection points P1, P2, P3.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_1.subtask_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_1_4 = (
        "Sub-task 4: Formulate the system of three equations based on the given triangle side lengths: |P1P2| = 200, |P2P3| = 240, and |P3P1| = 300. "
        "Express these distance constraints explicitly in terms of s and the angle parameters using the coordinates of P1, P2, and P3 derived previously. "
        "Output the full algebraic system ready for solving. This step translates the geometric problem into a concrete algebraic system."
    )
    cot_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_4 = []
    possible_thinkings_1_4 = []
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_sc_instruction_1_4,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_4, answer_1_4 = await cot_agents_1_4[i]([taskInfo, thinking_1_3, answer_1_3], cot_sc_instruction_1_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_4[i].id}, formulating system, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
        possible_answers_1_4.append(answer_1_4)
        possible_thinkings_1_4.append(thinking_1_4)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4(
        [taskInfo] + possible_answers_1_4 + possible_thinkings_1_4,
        "Sub-task 4: Synthesize and choose the most consistent algebraic system for the triangle side lengths.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_1.subtask_4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Solve the system of equations obtained in stage_1.subtask_4 for the unknowns: the hexagon side length s and any angle parameters introduced. "
        "Use analytic or numeric methods as appropriate, showing all intermediate steps, assumptions, and candidate solutions. "
        "Output all valid candidate solutions with detailed derivations. This step is critical to transition from symbolic formulation to concrete numeric results."
    )
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_4.content, answer_1_4.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_1, answer_2_1 = await cot_agents_2_1[i]([taskInfo, thinking_1_4, answer_1_4], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, solving system, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
        possible_answers_2_1.append(answer_2_1)
        possible_thinkings_2_1.append(thinking_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1(
        [taskInfo] + possible_answers_2_1 + possible_thinkings_2_1,
        "Sub-task 1: Synthesize and choose the most consistent candidate solutions for s and angles.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 5: ", sub_tasks[-1])

    reflect_inst_2_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Verify each candidate solution by substituting back into the original equations and geometric constraints. "
        "Check numeric consistency of the triangle side lengths, confirm the hexagon remains convex and equilateral, and ensure all pairs of opposite sides remain parallel. "
        "Eliminate any invalid or inconsistent solutions. If multiple candidates remain, initiate a structured debate among agents to reconcile conflicts and reach consensus on the unique valid solution. "
        "Provide a detailed verification report including numeric checks and geometric validations. " + reflect_inst_2_2
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1, thinking_1_4, answer_1_4, thinking_1_1, answer_1_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content, thinking_1_4.content, answer_1_4.content, thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "Reflexion | Debate"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, verifying candidates, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        critic_inst_2_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], critic_inst_2_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback_2_2.content}; answer: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining verification, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 6: ", sub_tasks[-1])

    debate_instr_2_3 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_2_3 = (
        "Sub-task 3: Present the final answer for the hexagon’s side length s, accompanied by a concise justification summarizing the derivation, verification, and conflict resolution process. "
        "Ensure the answer is unique, internally consistent, and fully justified by the preceding analysis. " + debate_instr_2_3
    )
    debate_agents_2_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_2_3 = self.max_round
    all_thinking_2_3 = [[] for _ in range(N_max_2_3)]
    all_answer_2_3 = [[] for _ in range(N_max_2_3)]
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": debate_instruction_2_3,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_3):
        for i, agent in enumerate(debate_agents_2_3):
            if r == 0:
                thinking_2_3, answer_2_3 = await agent([taskInfo, thinking_2_2, answer_2_2], debate_instruction_2_3, r, is_sub_task=True)
            else:
                input_infos_2_3 = [taskInfo, thinking_2_2, answer_2_2] + all_thinking_2_3[r-1] + all_answer_2_3[r-1]
                thinking_2_3, answer_2_3 = await agent(input_infos_2_3, debate_instruction_2_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, presenting final answer, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
            all_thinking_2_3[r].append(thinking_2_3)
            all_answer_2_3[r].append(answer_2_3)
    final_decision_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_3, answer_2_3 = await final_decision_agent_2_3(
        [taskInfo] + all_thinking_2_3[-1] + all_answer_2_3[-1],
        "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, calculating final answer, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_3, answer_2_3, sub_tasks, agents)
    return final_answer, logs
