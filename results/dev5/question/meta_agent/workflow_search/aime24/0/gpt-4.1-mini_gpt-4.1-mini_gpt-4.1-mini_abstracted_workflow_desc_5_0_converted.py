async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: aggregate_valid_combinations_and_derive_composite_measure
    # Subtask 1: Extract and formalize given information into equations
    cot_instruction_0_1 = (
        "Sub-task 1: Extract and formalize the given information into mathematical expressions. "
        "Represent the total time for walking plus coffee time at speeds s and s+2 as equations involving s and t. "
        "Convert all time units consistently (minutes to hours). Assume t is constant, walking speeds are constant, and total time includes walking plus coffee time. "
        "Given: distance=9 km, total time at speed s is 4 hours, total time at speed s+2 is 2 hours 24 minutes (2.4 hours). "
        "Write the two equations explicitly."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, extracting and formalizing equations, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Derive two equations and prepare for solving
    cot_instruction_0_2 = (
        "Sub-task 2: Derive two equations from the given conditions: 9/s + t/60 = 4 and 9/(s+2) + t/60 = 2.4. "
        "Isolate variables or express one variable in terms of the other to prepare for solving. "
        "Explain the algebraic manipulation clearly."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, deriving and preparing equations, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: infer_compute_parameters_from_composite_data
    # Subtask 3: Solve system of equations for s and t using Self-Consistency CoT
    cot_sc_instruction_1_3 = (
        "Sub-task 3: Solve the system of two equations derived previously to find values of s (walking speed) and t (coffee time in minutes). "
        "Use algebraic manipulation carefully, ensure positive and physically meaningful solutions. "
        "Use self-consistency by generating multiple solution attempts and select the most consistent one."
    )
    N_sc = self.max_sc
    cot_sc_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_1_3[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_3[i].id}, solving system, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_3.append(answer_i)
        possible_thinkings_1_3.append(thinking_i)

    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3(
        [taskInfo] + possible_answers_1_3 + possible_thinkings_1_3,
        "Sub-task 3: Synthesize and choose the most consistent and correct solutions for s and t.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent_1_3.id}, synthesizing solutions, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: aggregate_and_combine_values
    # Subtask 4: Compute total time for speed s + 0.5 km/h
    cot_instruction_2_4 = (
        "Sub-task 4: Using the solved values of s and t, compute the total time (walking + coffee) when Aya walks at speed s + 0.5 km/h. "
        "Calculate walking time as 9/(s + 0.5) hours, convert coffee time t from minutes to hours, sum them, and convert total time back to minutes. "
        "Provide the final answer as total number of minutes."
    )
    cot_agent_2_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_4 = {
        "subtask_id": "stage_2.subtask_4",
        "instruction": cot_instruction_2_4,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_4, answer_2_4 = await cot_agent_2_4([taskInfo, thinking_1_3, answer_1_3], cot_instruction_2_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_4.id}, computing total time for s+0.5, thinking: {thinking_2_4.content}; answer: {answer_2_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_2_4.content}; answer - {answer_2_4.content}")
    subtask_desc_2_4['response'] = {"thinking": thinking_2_4, "answer": answer_2_4}
    logs.append(subtask_desc_2_4)
    print("Step 4: ", sub_tasks[-1])

    # Subtask 5: Verify computed total time with Reflexion
    reflect_inst_2_5 = (
        "Sub-task 5: Given previous computations and answers, carefully verify the computed total time for s + 0.5 km/h. "
        "Check for consistency with original problem constraints, physical meaning, and correct formatting (in minutes). "
        "If any issues are found, refine the calculation."
    )
    cot_reflect_instruction_2_5 = (
        "Sub-task 5: Your problem is to find the total time for walking 9 km at speed s + 0.5 km/h including coffee time t. "
        + reflect_inst_2_5
    )
    cot_agent_2_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_5 = self.max_round
    cot_inputs_2_5 = [taskInfo, thinking_2_4, answer_2_4]
    subtask_desc_2_5 = {
        "subtask_id": "stage_2.subtask_5",
        "instruction": cot_reflect_instruction_2_5,
        "context": ["user query", thinking_2_4.content, answer_2_4.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_5, answer_2_5 = await cot_agent_2_5(cot_inputs_2_5, cot_reflect_instruction_2_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_5.id}, initial verification, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
    for i in range(N_max_2_5):
        feedback_2_5, correct_2_5 = await critic_agent_2_5(
            [taskInfo, thinking_2_5, answer_2_5],
            "Please review and provide limitations of the provided solution. If correct, output exactly 'True' in 'correct'.",
            i,
            is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent_2_5.id}, feedback round {i}, thinking: {feedback_2_5.content}; answer: {correct_2_5.content}")
        if correct_2_5.content.strip() == "True":
            break
        cot_inputs_2_5.extend([thinking_2_5, answer_2_5, feedback_2_5])
        thinking_2_5, answer_2_5 = await cot_agent_2_5(cot_inputs_2_5, cot_reflect_instruction_2_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_5.id}, refinement round {i+1}, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_2_5.content}; answer - {answer_2_5.content}")
    subtask_desc_2_5['response'] = {"thinking": thinking_2_5, "answer": answer_2_5}
    logs.append(subtask_desc_2_5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_5, answer_2_5, sub_tasks, agents)
    return final_answer, logs
