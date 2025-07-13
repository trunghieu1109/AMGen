async def forward_7(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Translate the given logarithmic equations into exponential form. "
        "From log_x(y^x) = 10 and log_y(x^{4y}) = 10, derive y^x = x^{10} and x^{4y} = y^{10}. "
        "Verify domain constraints x, y > 1 to ensure validity."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, translating logarithmic to exponential, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the exponential forms y^x = x^{10} and x^{4y} = y^{10}, "
        "combine and transform these equations to relate x and y more simply. "
        "Express variables in terms of natural logarithms and avoid premature approximations. "
        "Consider multiple cases for consistency."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, combining exponential equations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent and correct solution for the transformed system.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Analyze the system by setting a = ln x and b = ln y, "
        "and solve the resulting linear equations derived from the exponential system. "
        "Ensure solutions satisfy x, y > 1 and consider all possible solutions."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, solving linear system for ln x and ln y, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    cot_sc_instruction_3b = (
        "Sub-task 3b: Using self-consistency, verify and refine the solution for ln x and ln y, "
        "checking for multiple valid solutions and domain constraints."
    )
    cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3b = []
    possible_thinkings_3b = []
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", thinking3, answer3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3b, answer3b = await cot_agents_3b[i]([taskInfo, thinking3, answer3], cot_sc_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, verifying ln x and ln y solutions, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_answers_3b.append(answer3b)
        possible_thinkings_3b.append(thinking3b)
    thinking3b, answer3b = await final_decision_agent_2([taskInfo] + possible_answers_3b + possible_thinkings_3b, "Sub-task 3b: Finalize consistent ln x and ln y solutions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Compute the product xy from the values or expressions for x and y derived. "
        "Simplify the expression and verify by substituting back into the original equations. "
        "Use reflexion to ensure the solution is consistent and correct."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3b, answer3b],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_4 = [taskInfo, thinking3b, answer3b]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, computing product xy and verifying, thinking: {thinking4.content}; answer: {answer4.content}")
    N_max = self.max_round
    for i in range(N_max):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "Please review and provide limitations of the solution. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining solution, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
