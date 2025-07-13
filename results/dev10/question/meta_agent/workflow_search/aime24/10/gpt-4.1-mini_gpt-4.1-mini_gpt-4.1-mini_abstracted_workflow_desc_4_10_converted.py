async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Coordinate Setup
    cot_instruction_0 = (
        "Sub-task 1: Establish a coordinate system and assign coordinates to points A, B, C, D of rectangle ABCD and points E, F, G, H of rectangle EFGH based on the given side lengths (AB=107, BC=16 for ABCD; EF=184, FG=17 for EFGH). "
        "Position ABCD so that A is at the origin (0,0), AB lies along the positive x-axis, and BC along the positive y-axis, consistent with rectangle properties. "
        "Express coordinates of all points symbolically where possible, introducing variables only for unknown coordinates related to rectangle EFGH (especially the x-coordinate of E). "
        "Carefully preserve rectangle properties (right angles and equal opposite sides) without assuming orientations that contradict problem constraints. "
        "This setup will serve as the foundation for subsequent geometric constraint analysis. Avoid premature assumptions about the relative positions of rectangles beyond what is given."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, coordinate setup, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Stage 0 Subtask 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)

    # Stage 1 Subtask 1: Collinearity condition for D, E, C, F
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Using the coordinate setup from Stage 0, explicitly formulate the collinearity condition for points D, E, C, F. "
        "Determine the line equation on which these points lie (noting that D and C are fixed from ABCD, and E and F lie on the same line). "
        "Define the order and relative positions of D, E, C, and F on this line consistent with rectangle side lengths and orientations. "
        "Express coordinates of E and F in terms of variables introduced in Stage 0 and the line equation. Avoid assuming any order or position without algebraic justification."
    )
    N_sc = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, collinearity condition, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)

    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo] + possible_answers_1_1 + possible_thinkings_1_1,
        "Sub-task 1: Synthesize and choose the most consistent and correct solution for the collinearity condition of points D, E, C, F.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Subtask 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    # Stage 1 Subtask 2: Cyclic quadrilateral condition for A, D, H, G
    debate_instruction_1_2 = (
        "Sub-task 2: Formulate the cyclic quadrilateral condition for points A, D, H, G using the coordinates from Stage 0 and the expressions of H and G derived from E and F. "
        "Define variables explicitly: let E = (x_E,16), H = (x_E, 16 + s·17), G = (x_E + 184, 16 + s·17), with s in {+1, -1} for vertical offset sign. "
        "Using the determinant criterion for concyclicity or the equality of opposite angles, derive an explicit algebraic equation in terms of x_E and s. "
        "Solve this equation algebraically or symbolically to find numeric values for x_E and s. This step ensures the cyclic condition is rigorously and explicitly enforced rather than assumed or guessed. "
        "Avoid skipping algebraic solving or relying on verbal or geometric intuition alone. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_2 = self.max_round
    all_thinking_1_2 = [[] for _ in range(N_max_1_2)]
    all_answer_1_2 = [[] for _ in range(N_max_1_2)]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_1_2,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_2):
        for i, agent in enumerate(debate_agents_1_2):
            if r == 0:
                thinking_1_2, answer_1_2 = await agent([taskInfo, thinking_0, answer_0], debate_instruction_1_2, r, is_sub_task=True)
            else:
                input_infos_1_2 = [taskInfo, thinking_0, answer_0] + all_thinking_1_2[r-1] + all_answer_1_2[r-1]
                thinking_1_2, answer_1_2 = await agent(input_infos_1_2, debate_instruction_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, cyclic quadrilateral condition, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
            all_thinking_1_2[r].append(thinking_1_2)
            all_answer_1_2[r].append(answer_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo] + all_thinking_1_2[-1] + all_answer_1_2[-1],
        "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer for the cyclic quadrilateral condition.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Subtask 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # Stage 1 Subtask 3: Verification and reconciliation
    cot_sc_instruction_1_3 = (
        "Sub-task 3: Verify and reconcile the results of subtasks 1 and 2 by checking the compatibility of the collinearity and concyclicity conditions with the side lengths and orientations of rectangles ABCD and EFGH. "
        "Confirm that the derived values of x_E and s lead to consistent coordinates for points E, F, H, and G that satisfy rectangle properties (right angles, side lengths) and the collinearity and concyclicity constraints simultaneously. "
        "Identify if multiple solutions exist and exclude any that violate rectangle properties or given conditions. "
        "This subtask ensures uniqueness and correctness of the geometric configuration before proceeding."
    )
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_3, answer_1_3 = await cot_agents_1_3[i](
            [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2],
            cot_sc_instruction_1_3, is_sub_task=True
        )
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, verification and reconciliation, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
        possible_answers_1_3.append(answer_1_3)
        possible_thinkings_1_3.append(thinking_1_3)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3(
        [taskInfo] + possible_answers_1_3 + possible_thinkings_1_3,
        "Sub-task 3: Synthesize and choose the most consistent and correct solution for verification and reconciliation.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Subtask 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    # Stage 2 Subtask 1: Compute length CE
    debate_instruction_2_1 = (
        "Sub-task 1: Express the length CE explicitly in terms of the known side lengths and the solved coordinates from Stage 1. "
        "Using the established coordinates for points C and E (with x_E and s determined), compute CE as the Euclidean distance between these points. "
        "Simplify the expression to a numeric value. Avoid introducing new variables or assumptions at this stage; rely strictly on previously derived and verified data. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_2_1, answer_2_1 = await agent([taskInfo, thinking_1_3, answer_1_3], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking_1_3, answer_1_3] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_2_1, answer_2_1 = await agent(input_infos_2_1, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, compute length CE, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
            all_thinking_2_1[r].append(thinking_2_1)
            all_answer_2_1[r].append(answer_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1(
        [taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1],
        "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final numeric value for length CE.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 2 Subtask 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    # Stage 3 Subtask 1: Final aggregation and verification
    cot_sc_reflect_instruction_3_1 = (
        "Sub-task 1: Aggregate all derived numeric values and expressions from previous subtasks to finalize the length CE. "
        "Perform all necessary arithmetic calculations with precision, avoiding rounding errors unless justified. "
        "Verify the final answer against all problem constraints: rectangle properties, collinearity of D, E, C, F, and concyclicity of A, D, H, G. "
        "Provide a detailed verification summary confirming the consistency and correctness of the result. "
        "Return the final numeric value of CE along with this verification. This step prevents premature or unverified acceptance of the solution."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_1 = self.max_round
    cot_inputs_3_1 = [taskInfo, thinking_2_1, answer_2_1, thinking_1_3, answer_1_3, thinking_1_2, answer_1_2, thinking_1_1, answer_1_1, thinking_0, answer_0]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_reflect_instruction_3_1,
        "context": ["user query", thinking_2_1.content, answer_2_1.content, thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "CoT | SC_CoT | Reflexion"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1(cot_inputs_3_1, cot_sc_reflect_instruction_3_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, final aggregation and verification, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    critic_inst_3_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_3_1):
        feedback_3_1, correct_3_1 = await critic_agent_3_1([taskInfo, thinking_3_1, answer_3_1], critic_inst_3_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_1.id}, providing feedback, thinking: {feedback_3_1.content}; answer: {correct_3_1.content}")
        if correct_3_1.content == "True":
            break
        cot_inputs_3_1.extend([thinking_3_1, answer_3_1, feedback_3_1])
        thinking_3_1, answer_3_1 = await cot_agent_3_1(cot_inputs_3_1, cot_sc_reflect_instruction_3_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, refining final verification, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Stage 3 Subtask 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
