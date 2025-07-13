async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Formulate the mathematical model from the problem statement. "
        "Extract and clearly define all given quantities: distance (9 km), speeds (s, s+2, s+0.5 km/h), "
        "total times including coffee time t (4 hours and 2 hours 24 minutes). Convert all times to consistent units (hours), "
        "and express total time as walking time plus coffee time t (converted to hours). Establish two equations relating s and t based on the two given scenarios. "
        "Emphasize careful unit conversion and assumption that coffee time t is constant and independent of walking speed. "
        "Avoid assuming any unknowns beyond s and t.")

    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, formulating model, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {
        "thinking": thinking_0,
        "answer": answer_0
    }
    logs.append(subtask_desc_0)

    cot_sc_instruction_1 = (
        "Sub-task 2: Solve the system of two equations derived in Sub-task 1 to find numerical values of s (walking speed) and t (coffee time in hours). "
        "Use algebraic manipulation and substitution or elimination methods. Carefully handle units and verify that solutions are physically reasonable (s > 0, t >= 0). "
        "Avoid premature rounding to maintain accuracy.")

    N_sc = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    subtask_desc_1 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1, answer_1 = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, solving equations, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1.content)
        thinkingmapping_1[answer_1.content] = thinking_1
        answermapping_1[answer_1.content] = answer_1

    best_answer_1 = Counter(possible_answers_1).most_common(1)[0][0]
    thinking_1 = thinkingmapping_1[best_answer_1]
    answer_1 = answermapping_1[best_answer_1]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {
        "thinking": thinking_1,
        "answer": answer_1
    }
    logs.append(subtask_desc_1)

    cot_instruction_2 = (
        "Sub-task 3: Using the values of s and t obtained in Sub-task 2, calculate the total time (walking plus coffee) when walking at speed s + 0.5 km/h. "
        "Compute walking time as distance divided by (s + 0.5), add coffee time t (converted back to minutes if needed), and express the final answer in minutes. "
        "Ensure unit consistency and clarity in the final result.")

    reflect_inst = (
        "Sub-task 4: Verify the correctness of the computed total time by cross-checking with the original problem constraints and ensuring the answer is reasonable given the previous times. "
        "Provide a final answer with units and a brief explanation. If inconsistencies arise, revisit previous subtasks for correction.")

    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round

    subtask_desc_2 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking_2, answer_2 = await cot_agent_2([taskInfo, thinking_1, answer_1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, calculating total time at s+0.5, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {
        "thinking": thinking_2,
        "answer": answer_2
    }
    logs.append(subtask_desc_2)

    cot_reflect_instruction_3 = (
        "Sub-task 4: Your problem is to verify the correctness of the computed total time from Sub-task 3. "
        + reflect_inst)

    subtask_desc_3 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Reflexion"
    }

    cot_inputs_3 = [taskInfo, thinking_0, answer_0, thinking_1, answer_1, thinking_2, answer_2]

    thinking_3, answer_3 = await cot_agent_2(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, verifying total time, thinking: {thinking_3.content}; answer: {answer_3.content}")

    for i in range(N_max_reflect):
        feedback, correct = await critic_agent_2([taskInfo, thinking_3, answer_3],
                                               "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_3.extend([thinking_3, answer_3, feedback])
        thinking_3, answer_3 = await cot_agent_2(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining total time verification, thinking: {thinking_3.content}; answer: {answer_3.content}")

    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {
        "thinking": thinking_3,
        "answer": answer_3
    }
    logs.append(subtask_desc_3)

    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    return final_answer, logs
