async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Derive coordinate or vector representations for rectangles ABCD and EFGH using the given side lengths and rectangle properties. "
        "Assign coordinates to points A, B, C, D, E, F, G, H consistent with rectangle definitions and given side lengths (AB=107, BC=16 for ABCD; EF=184, FG=17 for EFGH). "
        "Incorporate the fact that ABCD and EFGH are rectangles (right angles, opposite sides equal). Avoid assuming arbitrary orientations without justification; consider standard labeling and orientation to simplify calculations.")
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_0_1}")
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, deriving coordinates, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_0_3 = (
        "Sub-task 3: Validate and reconcile the coordinate assignments and constraints to ensure consistency. "
        "Check that the side lengths, rectangle properties, collinearity, and concyclicity conditions can be simultaneously satisfied. "
        "Adjust coordinate assignments if necessary to maintain all conditions. This step ensures a valid geometric model for further analysis.")
    N_sc = self.max_sc
    cot_sc_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before SC-CoT agent calls: {subtask_desc_0_3}")
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_0_3[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_3[i].id}, validating coordinates, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_answers_0_3 + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent and correct coordinate validation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Identify and select the exact order and relative positions of points D, E, C, F on the collinear line, consistent with the rectangles' geometry and given side lengths. "
        "Determine the segment lengths along this line where possible, using rectangle side lengths and the collinearity condition. Verify uniqueness or possible configurations that satisfy all constraints.")
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before SC-CoT agent calls: {subtask_desc_1_1}")
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_1_1[i]([taskInfo, thinking_0_3, answer_0_3], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, selecting collinear points order, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent order of points D, E, C, F.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_1_2 = (
        "Sub-task 2: Analyze the cyclic quadrilateral formed by points A, D, H, G. Use properties of cyclic quadrilaterals (e.g., opposite angles sum to 180°, power of a point, or circle equations) to relate the positions of points H and G to the known points and side lengths. "
        "Verify which points correspond to which vertices in EFGH to maintain consistency with given side lengths EF=184 and FG=17.")
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_1_2}")
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_0_3, answer_0_3], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, analyzing cyclic quadrilateral, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_1_3 = (
        "Sub-task 3: Verify that the selected points and their positions satisfy all rectangle properties, collinearity, and concyclicity constraints simultaneously. "
        "Confirm that the configuration is geometrically valid and consistent with the problem statement. This verification ensures the correctness of the geometric model before proceeding to numeric decomposition.")
    cot_sc_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before SC-CoT agent calls: {subtask_desc_1_3}")
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_1_3[i]([taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_3[i].id}, verifying configuration, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_3.append(answer_i)
        possible_thinkings_1_3.append(thinking_i)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_answers_1_3 + possible_thinkings_1_3, "Sub-task 3: Synthesize and confirm geometric validity.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 5: ", sub_tasks[-1])

    debate_instruction_2_1 = (
        "Sub-task 1: Decompose the length CE into components based on the established coordinate system and geometric relations. "
        "Express CE as a sum or difference of known segment lengths or as a function of variables constrained by the rectangle and circle properties. Simplify these expressions to minimal form, using ratios or partitions if applicable.")
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_rounds)]
    all_answer_2_1 = [[] for _ in range(N_rounds)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Debate"
    }
    print(f"Logging before Debate agent calls: {subtask_desc_2_1}")
    for r in range(N_rounds):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_3, answer_1_3], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_3, answer_1_3] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, decomposing CE, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_1[r].append(thinking_i)
            all_answer_2_1[r].append(answer_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 1: Synthesize and simplify CE decomposition.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 6: ", sub_tasks[-1])

    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Simplify and compute intermediate lengths or ratios derived from the cyclic quadrilateral and collinearity conditions that contribute to the length CE. "
        "Use algebraic manipulation and geometric theorems (e.g., power of a point, similarity, Pythagoras) to reduce complexity and prepare for final aggregation.")
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before Reflexion agent call: {subtask_desc_2_2}")
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, simplifying CE components, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_reflect):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], "Please review and provide the limitations of provided solutions. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining simplification, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction_3_1 = (
        "Sub-task 1: Aggregate the simplified components and intermediate values to compute the final length of segment CE. "
        "Apply arithmetic operations and verify the result against all constraints. Provide the final numeric value of CE with appropriate justification.")
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_3_1}")
    thinking_3_1, answer_3_1 = await cot_agent_3_1([taskInfo, thinking_2_2, answer_2_2], cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, aggregating final CE length, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 8: ", sub_tasks[-1])

    cot_sc_instruction_3_2 = (
        "Sub-task 2: Verify the computed length CE by cross-checking with the original geometric constraints (rectangle properties, collinearity, concyclicity) and given side lengths. "
        "Confirm that the solution is consistent and unique. If inconsistencies arise, revisit previous subtasks for correction.")
    cot_sc_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3_2 = []
    possible_thinkings_3_2 = []
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_sc_instruction_3_2,
        "context": ["user query", thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before SC-CoT agent calls: {subtask_desc_3_2}")
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_3_2[i]([taskInfo, thinking_3_1, answer_3_1], cot_sc_instruction_3_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3_2[i].id}, verifying final CE length, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3_2.append(answer_i)
        possible_thinkings_3_2.append(thinking_i)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_2, answer_3_2 = await final_decision_agent_3_2([taskInfo] + possible_answers_3_2 + possible_thinkings_3_2, "Sub-task 2: Synthesize and confirm final CE length correctness.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
