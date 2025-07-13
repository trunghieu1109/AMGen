async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_N = self.max_sc

    # Stage 1: Define hexagon properties and line equations, compute intersections

    # Subtask 1: Define geometric properties of the hexagon with fixed directions
    cot_sc_instruction_1 = (
        "Sub-task 1: Formally define the geometric properties of the convex equilateral hexagon ABCDEF. "
        "Explicitly state that all sides have equal length s and that opposite sides are parallel with directions fixed at 0°, 120°, and 240° for sides AB, CD, and EF respectively. "
        "Explain the implications of these fixed directions on the hexagon's structure. Do not perform numeric computations or assumptions beyond these constraints. "
        "Provide the output as a structured JSON object with keys: 'side_length_variable', 'directions_degrees', 'parallel_pairs', and 'implications'."
    )

    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(cot_sc_N)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }

    for i in range(cot_sc_N):
        thinking1, answer1 = await cot_sc_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, defining hexagon properties, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)

    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1, "Sub-task 1: Synthesize and choose the most consistent definition of hexagon properties.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    # Parse structured output from answer1
    import json
    try:
        hexagon_props = json.loads(answer1.content)
    except Exception:
        hexagon_props = {}

    # Subtask 2: Represent lines AB, CD, EF as parametric equations with fixed directions
    cot_sc_instruction_2 = (
        "Sub-task 2: Using the fixed directions 0°, 120°, and 240° for sides AB, CD, and EF respectively, "
        "represent the lines containing these sides as parametric equations in a coordinate system. "
        "Express the equations in terms of the unknown side length s and a chosen origin point (e.g., vertex A at (0,0)). "
        "Provide explicit parametric or normal form equations for lines AB, CD, and EF, and output as a JSON object with keys: 'line_AB', 'line_CD', 'line_EF', each containing the parametric form coefficients or normal form coefficients in symbolic form involving s. "
        "Do not assume anything about the triangle formed by their intersections at this stage."
    )

    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(cot_sc_N)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }

    for i in range(cot_sc_N):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo, thinking1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, representing lines, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent line equations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    try:
        line_eqs = json.loads(answer2.content)
    except Exception:
        line_eqs = {}

    # Subtask 3: Compute intersection points P=AB∩CD, Q=CD∩EF, R=EF∩AB symbolically
    cot_sc_instruction_3 = (
        "Sub-task 3: Using the parametric or normal form line equations of AB, CD, and EF from Sub-task 2, "
        "compute the intersection points P = AB∩CD, Q = CD∩EF, and R = EF∩AB symbolically. "
        "Express each intersection point as coordinates in terms of the unknown side length s and any fixed parameters. "
        "Output the points as a JSON object with keys 'P', 'Q', 'R', each containing coordinate pairs (x, y) in symbolic form. "
        "Do not relate these points to the triangle side lengths yet."
    )

    cot_sc_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(cot_sc_N)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking2.content],
        "agent_collaboration": "SC_CoT"
    }

    for i in range(cot_sc_N):
        thinking3, answer3 = await cot_sc_agents_3[i]([taskInfo, thinking2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3[i].id}, computing intersections, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)

    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent intersection points.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    try:
        intersections = json.loads(answer3.content)
    except Exception:
        intersections = {}

    # Subtask 4: Derive exact symbolic formulas for distances |PQ|, |QR|, |RP| in terms of s
    debate_instruction_4 = (
        "Sub-task 4: Using the intersection points P, Q, R from Sub-task 3, derive exact symbolic formulas for the distances |PQ|, |QR|, and |RP|. "
        "Express each distance solely in terms of the hexagon side length s and known constants (including fixed directions). "
        "Provide the formulas as a JSON object with keys 'PQ', 'QR', 'RP', each containing the symbolic distance formula. "
        "Avoid arbitrary proportionality constants or assumptions. Ensure formulas are ready to be equated to the given triangle side lengths 200, 240, and 300. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )

    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3] + all_thinking_4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, deriving distance formulas, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_4[r].append(thinking4)
            all_answer_4[r].append(answer4)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1], "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide final exact distance formulas.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    try:
        distance_formulas = json.loads(answer4.content)
    except Exception:
        distance_formulas = {}

    # Stage 2: Formulate system and solve numerically

    # Subtask 1: Formulate system of equations setting distances equal to 200, 240, 300
    debate_instruction_5_1 = (
        "Sub-task 1: Formulate the system of equations by setting the derived distance formulas |PQ|, |QR|, and |RP| equal to the given triangle side lengths 200, 240, and 300 respectively. "
        "Explicitly write down the system in terms of the single unknown s. Confirm the system is well-constrained given the fixed directions and hexagon properties. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )

    debate_agents_5_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_5_1 = self.max_round
    all_thinking_5_1 = [[] for _ in range(N_max_5_1)]
    all_answer_5_1 = [[] for _ in range(N_max_5_1)]
    subtask_desc_5_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_5_1,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_5_1):
        for i, agent in enumerate(debate_agents_5_1):
            if r == 0:
                thinking5_1, answer5_1 = await agent([taskInfo, thinking4], debate_instruction_5_1, r, is_sub_task=True)
            else:
                input_infos_5_1 = [taskInfo, thinking4] + all_thinking_5_1[r-1]
                thinking5_1, answer5_1 = await agent(input_infos_5_1, debate_instruction_5_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, formulating system, thinking: {thinking5_1.content}; answer: {answer5_1.content}")
            all_thinking_5_1[r].append(thinking5_1)
            all_answer_5_1[r].append(answer5_1)

    final_decision_agent_5_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5_1, answer5_1 = await final_decision_agent_5_1([taskInfo] + all_thinking_5_1[-1], "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide the final system of equations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking5_1.content}; answer - {answer5_1.content}")
    subtask_desc_5_1['response'] = {"thinking": thinking5_1, "answer": answer5_1}
    logs.append(subtask_desc_5_1)
    print("Step 5.1: ", sub_tasks[-1])

    # Subtask 2: Numerically solve the system for s
    debate_instruction_5_2 = (
        "Sub-task 2: Numerically solve the system of equations from Sub-task 1 to find the exact value of the hexagon side length s. "
        "Use appropriate numeric or symbolic methods ensuring convergence and correctness. Provide the numeric value of s with sufficient precision. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )

    debate_agents_5_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_5_2 = self.max_round
    all_thinking_5_2 = [[] for _ in range(N_max_5_2)]
    all_answer_5_2 = [[] for _ in range(N_max_5_2)]
    subtask_desc_5_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": debate_instruction_5_2,
        "context": ["user query", thinking5_1.content],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_5_2):
        for i, agent in enumerate(debate_agents_5_2):
            if r == 0:
                thinking5_2, answer5_2 = await agent([taskInfo, thinking5_1], debate_instruction_5_2, r, is_sub_task=True)
            else:
                input_infos_5_2 = [taskInfo, thinking5_1] + all_thinking_5_2[r-1]
                thinking5_2, answer5_2 = await agent(input_infos_5_2, debate_instruction_5_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, solving numerically, thinking: {thinking5_2.content}; answer: {answer5_2.content}")
            all_thinking_5_2[r].append(thinking5_2)
            all_answer_5_2[r].append(answer5_2)

    final_decision_agent_5_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5_2, answer5_2 = await final_decision_agent_5_2([taskInfo] + all_thinking_5_2[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide the numeric solution for s.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking5_2.content}; answer - {answer5_2.content}")
    subtask_desc_5_2['response'] = {"thinking": thinking5_2, "answer": answer5_2}
    logs.append(subtask_desc_5_2)
    print("Step 5.2: ", sub_tasks[-1])

    # Stage 2 Subtask 3: Reflexion and validation
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = (
        "Sub-task 3: Validate the computed side length s by substituting it back into the distance formulas from Sub-task 4 and verifying that the resulting triangle side lengths match 200, 240, and 300 within acceptable numeric tolerance. "
        "Additionally, verify that the hexagon remains convex and that all geometric constraints (parallelism, fixed directions, equilateral sides) hold true. "
        "If inconsistencies arise, report and suggest iteration or refinement. "
        + reflect_inst
    )

    cot_agent_reflect = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_reflect = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round

    cot_inputs_reflect = [taskInfo, thinking5_2, thinking4]
    subtask_desc_5_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_reflect_instruction,
        "context": ["user query", thinking5_2.content, thinking4.content],
        "agent_collaboration": "Reflexion"
    }

    thinking5_3, answer5_3 = await cot_agent_reflect(cot_inputs_reflect, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, validating solution, thinking: {thinking5_3.content}; answer: {answer5_3.content}")

    for i in range(N_max_reflect):
        feedback, correct = await critic_agent_reflect([taskInfo, thinking5_3], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_reflect.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_reflect.extend([thinking5_3, feedback])
        thinking5_3, answer5_3 = await cot_agent_reflect(cot_inputs_reflect, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, refining solution, thinking: {thinking5_3.content}; answer: {answer5_3.content}")

    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking5_3.content}; answer - {answer5_3.content}")
    subtask_desc_5_3['response'] = {"thinking": thinking5_3, "answer": answer5_3}
    logs.append(subtask_desc_5_3)
    print("Step 5.3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5_3, answer5_3, sub_tasks, agents)
    return final_answer, logs
