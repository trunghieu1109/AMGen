async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Formalize the given problem by representing the complex number z as z = 4e^{iθ}, "
        "where θ ∈ [0, 2π). Express the given expression (75 + 117i)z + (96 + 144i)/z in terms of θ, "
        "using properties of complex numbers, including the reciprocal and modulus constraints. "
        "Extract the real part of the expression explicitly as a function of θ. "
        "Carefully consider the implications of the modulus constraint |z|=4 and avoid assumptions that restrict θ beyond the given domain."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, formalizing expression, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {
        "thinking": thinking_0,
        "answer": answer_0
    }
    logs.append(subtask_desc_0)

    cot_sc_instruction_1 = (
        "Sub-task 2: Combine and transform the expression obtained in stage_0 to a simplified trigonometric form. "
        "This involves expanding the terms, separating real and imaginary parts, and expressing the real part as a sum of cosine and sine functions with coefficients. "
        "Aggregate constants and simplify the expression so it is suitable for optimization over θ. "
        "Avoid algebraic errors and ensure the transformations preserve equivalence."
    )
    N_sc = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1, answer_1 = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, simplifying expression, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1)
        possible_thinkings_1.append(thinking_1)

    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1 = "Given all the above thinking and answers, find the most consistent and correct simplified trigonometric form for the real part expression."
    thinking_1, answer_1 = await final_decision_agent_1([taskInfo] + possible_answers_1 + possible_thinkings_1, "Sub-task 2: Synthesize and choose the most consistent answer for expression simplification." + final_instr_1, is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing simplified expression, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {
        "thinking": thinking_1,
        "answer": answer_1
    }
    logs.append(subtask_desc_1)

    cot_reflect_instruction_2 = (
        "Sub-task 3: Determine the maximum possible value of the real part expression derived in stage_1 by optimizing over θ ∈ [0, 2π). "
        "Use trigonometric identities or vector magnitude arguments to find the maximum value of the sum of sine and cosine terms. "
        "Compute the explicit maximum value and identify the corresponding θ if needed. "
        "Verify the result for correctness and provide the final answer for the largest possible real part. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_2 = [taskInfo, thinking_1, answer_1]
    subtask_desc_2 = {
        "subtask_id": "subtask_1",
        "instruction": cot_reflect_instruction_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2, answer_2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, optimizing expression, thinking: {thinking_2.content}; answer: {answer_2.content}")
    for i in range(N_max):
        critic_inst_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback_2, correct_2 = await critic_agent_2([taskInfo, thinking_2, answer_2], "Please review and provide the limitations of provided solutions." + critic_inst_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback_2.content}; answer: {correct_2.content}")
        if correct_2.content == "True":
            break
        cot_inputs_2.extend([thinking_2, answer_2, feedback_2])
        thinking_2, answer_2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining optimization, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {
        "thinking": thinking_2,
        "answer": answer_2
    }
    logs.append(subtask_desc_2)

    final_answer = await self.make_final_answer(thinking_2, answer_2, sub_tasks, agents)
    return final_answer, logs
