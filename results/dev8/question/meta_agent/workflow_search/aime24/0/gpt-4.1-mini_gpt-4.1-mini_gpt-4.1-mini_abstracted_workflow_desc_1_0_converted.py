async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_0 = (
        "Sub-task 0_0: Formulate the system of equations from the given data. "
        "Given: distance = 9 km, speeds s and s+2 km/h, total times 4 hours and 2.4 hours including coffee time t (in minutes). "
        "Express the two equations: 4 = 9/s + t/60 and 2.4 = 9/(s+2) + t/60. "
        "Do not solve yet, just set up the equations clearly with correct units and assumptions."
    )
    cot_agent_0_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_0 = {
        "subtask_id": "subtask_0",
        "instruction": cot_instruction_0_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_0, answer_0_0 = await cot_agent_0_0([taskInfo], cot_instruction_0_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_0.id}, formulating equations, thinking: {thinking_0_0.content}; answer: {answer_0_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0_0.content}; answer - {answer_0_0.content}")
    subtask_desc_0_0['response'] = {"thinking": thinking_0_0, "answer": answer_0_0}
    logs.append(subtask_desc_0_0)

    cot_sc_instruction_0_1 = (
        "Sub-task 0_1: Solve the system of equations from Sub-task 0_0 to find numerical values of s (walking speed) and t (coffee time in minutes). "
        "Use algebraic manipulation and verify solutions are positive and reasonable. "
        "Maintain unit consistency and avoid premature rounding."
    )
    N = self.max_sc
    cot_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query", thinking_0_0.content, answer_0_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_i, answer_i = await cot_agents_0_1[i]([taskInfo, thinking_0_0, answer_0_0], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_1[i].id}, solving equations, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_1.append(answer_i)
        possible_thinkings_0_1.append(thinking_i)

    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1(
        [taskInfo] + possible_answers_0_1 + possible_thinkings_0_1,
        "Sub-task 0_1: Synthesize and choose the most consistent and correct solutions for s and t.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_instruction_1_2 = (
        "Sub-task 1_2: Using the values of s and t obtained in Sub-task 0_1, compute the total time (walking + coffee) when Aya walks at speed s + 0.5 km/h. "
        "Calculate walking time as 9/(s+0.5) hours, add coffee time t (converted to hours), then convert total time to minutes. "
        "Ensure unit conversions are correct and coffee time is included properly."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, computing total time at s+0.5, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    cot_reflect_instruction_1_3 = (
        "Sub-task 1_3: Verify the computed total time for speed s + 0.5 km/h by cross-checking calculations and ensuring consistency with problem constraints. "
        "Reflect on the reasonableness of the answer given previous total times (4 hours and 2.4 hours). "
        "Provide the final answer in minutes clearly and check no component is overlooked."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking_0_1, answer_0_1, thinking_1_2, answer_1_2]
    subtask_desc_1_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, verifying total time, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_1_3(
            [taskInfo, thinking_1_3, answer_1_3],
            "Please review and provide limitations of the provided solution. If absolutely correct, output exactly 'True' in 'correct'.",
            i, is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent_1_3.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, answer_1_3, feedback])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining solution, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    final_answer = await self.make_final_answer(thinking_1_3, answer_1_3, sub_tasks, agents)
    return final_answer, logs
