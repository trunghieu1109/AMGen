async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Derive a suitable representation of the complex number z and the given expression. "
        "Express z in polar form as z = 4 * exp(i * theta) with theta in [0, 2*pi). "
        "Rewrite the expression (75 + 117i) * z + (96 + 144i) / z in terms of theta, separating real and imaginary parts. "
        "Validate that this representation correctly captures the problem constraints and is suitable for further analysis."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, deriving polar form and expression, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {
        "thinking": thinking_0,
        "answer": answer_0
    }
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    cot_instruction_1 = (
        "Sub-task 1: Combine and transform the expression derived in Stage 0 into a single trigonometric expression representing the real part as a function of theta. "
        "Expand the products, use Euler's formula, and group cosine and sine terms. "
        "Obtain a simplified formula for the real part that can be optimized over theta. "
        "Ensure the transformation respects the fixed modulus constraint and correctly aggregates all terms."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "CoT"
    }
    thinking_1, answer_1 = await cot_agent_1([taskInfo, thinking_0, answer_0], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, transforming expression to trig form, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {
        "thinking": thinking_1,
        "answer": answer_1
    }
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze the trigonometric expression obtained in Stage 1 to identify parameters (amplitudes and phase shifts) that allow rewriting the real part as a single cosine function with a phase shift. "
        "Compute these parameters explicitly. Then, determine the value of theta that maximizes this cosine function, thereby maximizing the real part of the original expression. "
        "Carefully handle the interplay between terms and ensure the solution respects the domain of theta."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_2, answer_2 = await cot_agents_2[i]([taskInfo, thinking_1, answer_1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing trig expression and maximizing, thinking: {thinking_2.content}; answer: {answer_2.content}")
        possible_answers_2.append(answer_2.content)
        thinkingmapping_2[answer_2.content] = thinking_2
        answermapping_2[answer_2.content] = answer_2
    best_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking_2 = thinkingmapping_2[best_answer_2]
    answer_2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {
        "thinking": thinking_2,
        "answer": answer_2
    }
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_reflect_instruction_3 = (
        "Sub-task 3: Select the maximum real part value found in Stage 2 and verify its correctness by substituting back into the original expression or by alternative reasoning (e.g., geometric interpretation or numerical check). "
        "Confirm that the maximum is achievable given the constraints and that no larger value exists. Provide the final answer along with a brief justification or verification summary."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_3 = [taskInfo, thinking_2, answer_2]
    subtask_desc_3 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", thinking_2.content, answer_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, verifying max real part, thinking: {thinking_3.content}; answer: {answer_3.content}")
    critic_inst_3 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max):
        feedback_3, correct_3 = await critic_agent_3([taskInfo, thinking_3, answer_3], "Please review and provide the limitations of provided solutions" + critic_inst_3, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback_3.content}; answer: {correct_3.content}")
        if correct_3.content == "True":
            break
        cot_inputs_3.extend([thinking_3, answer_3, feedback_3])
        thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining valid scenarios, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {
        "thinking": thinking_3,
        "answer": answer_3
    }
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    return final_answer, logs
