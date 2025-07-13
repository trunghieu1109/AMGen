async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 0: Extract and formalize the given information into algebraic expressions. "
        "Define variables s (walking speed in km/h) and t (coffee shop time in minutes). "
        "Convert all times to hours. Formulate two equations based on total time = walking time + coffee shop time, "
        "using the given speeds and total times. Avoid assumptions before solving."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_0",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, extracting and formalizing equations, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)

    cot_sc_instruction_1 = (
        "Sub-task 1: Solve the system of two equations derived in Sub-task 0 to find s (walking speed) and t (coffee shop time). "
        "Use algebraic manipulation or substitution methods. Verify solutions are positive and reasonable. Avoid premature rounding."
    )
    N = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", "thinking of subtask 0", "answer of subtask 0"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_1, answer_1 = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, solving system of equations, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1)
        possible_thinkings_1.append(thinking_1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1([taskInfo] + possible_answers_1 + possible_thinkings_1, 
                                                      "Sub-task 1: Synthesize and choose the most consistent and correct solutions for s and t.", 
                                                      is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)

    cot_instruction_2 = (
        "Sub-task 2: Using the computed values of s and t from Sub-task 1, calculate the total time (walking + coffee shop) when Aya walks at speed s + 0.5 km/h. "
        "Convert total time into minutes. Combine walking time and coffee shop time carefully with consistent units."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking_2, answer_2 = await cot_agent_2([taskInfo, thinking_1, answer_1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, calculating total time at s+0.5, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)

    cot_reflect_instruction_3 = (
        "Sub-task 3: Verify the final computed total time by cross-checking with original problem constraints and ensuring consistency with times at different speeds. "
        "Reflect on reasonableness and confirm no contradictions."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_3 = [taskInfo, thinking_2, answer_2]
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, verifying final time, thinking: {thinking_3.content}; answer: {answer_3.content}")
    for i in range(N_max):
        feedback_3, correct_3 = await critic_agent_3([taskInfo, thinking_3, answer_3], 
                                                   "Please review and provide limitations of the solution. If correct, output exactly 'True' in 'correct'.", 
                                                   i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback: {feedback_3.content}; correct: {correct_3.content}")
        if correct_3.content == "True":
            break
        cot_inputs_3.extend([thinking_3, answer_3, feedback_3])
        thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining solution, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)

    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    for i, step in enumerate(sub_tasks, 1):
        print(f"Step {i}: ", step)
    return final_answer, logs
