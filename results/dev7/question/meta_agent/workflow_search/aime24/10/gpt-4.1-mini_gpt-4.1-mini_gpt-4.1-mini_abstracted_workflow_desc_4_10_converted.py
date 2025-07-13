async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Derive coordinate or vector representations for rectangles ABCD and EFGH based on given side lengths and rectangle properties. "
        "Assign a coordinate system placing ABCD aligned with axes, with A at origin, AB along x-axis, and BC along y-axis. "
        "Use AB=107, BC=16 for ABCD; EF=184, FG=17 for EFGH. "
        "Express coordinates of all points A,B,C,D,E,F,G,H accordingly. "
        "Avoid arbitrary orientations; choose placements that simplify the problem."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, deriving coordinates, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Formally represent the collinearity condition of points D, E, C, F using the coordinates from Sub-task 1. "
        "Express the linear equation or vector relation that these four points satisfy. "
        "Consider possible orders of points on the line and deduce constraints on E and F positions."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, representing collinearity, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Express the concyclicity condition of points A, D, H, G using coordinates from Sub-task 1. "
        "Derive the equation of the circle passing through these points or use cyclic quadrilateral properties to relate points from both rectangles. "
        "Avoid assuming circle center or radius without derivation."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, expressing concyclicity, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Identify and verify the relative positions and orientations of rectangles ABCD and EFGH under constraints from Stage 0. "
        "Confirm the order of points D, E, C, F on the line and placement of EFGH relative to ABCD. "
        "Use collinearity and concyclicity conditions to narrow down configurations. "
        "Verify side lengths and right angles consistency."
    )
    N_sc = self.max_sc
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content, thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_sc_agents_1_1[i]([taskInfo, thinking_0_2, answer_0_2, thinking_0_3, answer_0_3], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, verifying configurations, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent configuration for rectangles and points order.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Enumerate and select geometric elements satisfying all constraints including side lengths, collinearity, and concyclicity. "
        "Determine unknown segment lengths or coordinates to express CE. "
        "Verify uniqueness or existence of solutions."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, enumerating geometric elements, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    debate_instruction_2_1 = (
        "Sub-task 1: Decompose segment CE into components based on linear arrangement of points D, E, C, F and known side lengths. "
        "Express CE as sum or difference of known or derived segments. Simplify expressions. "
        "Consider segment order and avoid unnecessary complexity."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_rounds_2_1)]
    all_answer_2_1 = [[] for _ in range(N_rounds_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_2_1, answer_2_1 = await agent([taskInfo, thinking_1_2, answer_1_2], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking_1_2, answer_1_2] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_2_1, answer_2_1 = await agent(input_infos_2_1, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, decomposing CE, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
            all_thinking_2_1[r].append(thinking_2_1)
            all_answer_2_1[r].append(answer_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 1: Finalize decomposition and expression of CE.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    reflect_instruction_2_2 = (
        "Sub-task 2: Simplify and compute intermediate values related to the circle condition that influence CE. "
        "Use concyclicity condition to relate segments involving A, D, H, G and constrain CE. "
        "Keep expressions exact or symbolic until final aggregation."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_0_3, answer_0_3, thinking_1_2, answer_1_2]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": reflect_instruction_2_2,
        "context": ["user query", thinking_0_3.content, answer_0_3.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, simplifying circle constraints, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], "Please review and provide limitations of the solution. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback: {feedback_2_2.content}; correct: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining solution, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    cot_instruction_3_1 = (
        "Sub-task 1: Aggregate all derived expressions and numeric values from previous subtasks to compute the exact length of CE. "
        "Combine segment decompositions and circle constraints to solve for CE numerically or symbolically. "
        "Verify solution consistency with all given conditions and provide final answer."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", thinking_2_1.content, answer_2_1.content, thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1([taskInfo, thinking_2_1, answer_2_1, thinking_2_2, answer_2_2], cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, aggregating and computing CE, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    for i, step in enumerate(sub_tasks, 1):
        print(f"Step {i}: ", step)
    return final_answer, logs
