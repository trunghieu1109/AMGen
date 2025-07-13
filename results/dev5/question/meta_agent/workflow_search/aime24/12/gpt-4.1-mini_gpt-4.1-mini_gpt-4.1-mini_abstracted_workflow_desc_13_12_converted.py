async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Derive the parametric form of z given |z|=4, i.e., express z as 4e^{iθ} for θ in [0, 2π). "
        "Rewrite the expression (75 + 117i)z + (96 + 144i)/z in terms of θ. "
        "Validate algebraic manipulations and ensure the expression is suitable for extracting the real part. "
        "Avoid assumptions beyond θ's domain and ensure no division by zero occurs."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, deriving parametric form and rewriting expression, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_0 = (
        "Sub-task 2: Based on the parametric form and rewritten expression from Sub-task 1, "
        "consider multiple independent calculations to confirm the correctness and consistency of the expression in terms of θ. "
        "Check for algebraic errors or alternative forms."
    )
    N_sc = self.max_sc
    cot_agents_sc_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc_sc_0 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_sc_0, answer_sc_0 = await cot_agents_sc_0[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc_0[i].id}, verifying parametric expression, thinking: {thinking_sc_0.content}; answer: {answer_sc_0.content}")
        possible_answers_0.append(answer_sc_0)
        possible_thinkings_0.append(thinking_sc_0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_final, answer_0_final = await final_decision_agent_0([taskInfo] + possible_answers_0 + possible_thinkings_0, "Sub-task 2: Synthesize and confirm the most consistent parametric expression for the given problem.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0 verification output: thinking - {thinking_0_final.content}; answer - {answer_0_final.content}")
    subtask_desc_sc_0['response'] = {"thinking": thinking_0_final, "answer": answer_0_final}
    logs.append(subtask_desc_sc_0)
    print("Step 0 verification: ", sub_tasks[-1])

    reflexion_instruction_1 = (
        "Sub-task 1: From the parametric expression confirmed in Stage 0, separate the real and imaginary parts explicitly. "
        "Combine terms to form a single trigonometric expression involving cos(θ) and sin(θ) representing the real part. "
        "Emphasize algebraic simplification and correctness, avoiding skipped steps."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_reflect = self.max_round
    cot_inputs_1 = [taskInfo, thinking_0_final, answer_0_final]
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": reflexion_instruction_1,
        "context": ["user query", thinking_0_final.content, answer_0_final.content],
        "agent_collaboration": "Reflexion | CoT"
    }
    thinking_1, answer_1 = await cot_agent_1(cot_inputs_1, reflexion_instruction_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1.id}, separating real and imaginary parts, thinking: {thinking_1.content}; answer: {answer_1.content}")
    for i in range(N_reflect):
        feedback_1, correct_1 = await critic_agent_1([taskInfo, thinking_1, answer_1], "Please review and provide limitations of the solution. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1.id}, feedback: {feedback_1.content}; correct: {correct_1.content}")
        if correct_1.content == "True":
            break
        cot_inputs_1.extend([thinking_1, answer_1, feedback_1])
        thinking_1, answer_1 = await cot_agent_1(cot_inputs_1, reflexion_instruction_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1.id}, refining expression, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 1: Analyze the trigonometric expression from Stage 1 to identify parameters (amplitudes and phase shifts) "
        "that allow rewriting it as R cos(θ - φ) + constant if applicable. Compute these parameters explicitly. "
        "Handle coefficients carefully and apply trigonometric identities correctly."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1, answer_1], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, computing parameters for trig expression, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Determine the maximum value of the real part by maximizing the trigonometric expression over θ in [0, 2π). "
        "Identify the θ that achieves this maximum. Avoid assuming multiple maxima without verification."
    )
    cot_agents_sc_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_sc_2_2, answer_sc_2_2 = await cot_agents_sc_2_2[i]([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc_2_2[i].id}, maximizing trig expression, thinking: {thinking_sc_2_2.content}; answer: {answer_sc_2_2.content}")
        possible_answers_2_2.append(answer_sc_2_2)
        possible_thinkings_2_2.append(thinking_sc_2_2)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_2_2 + possible_thinkings_2_2, "Sub-task 2.2: Synthesize and confirm the maximum value and maximizing θ.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Select the value(s) of θ that maximize the real part and verify these values satisfy the original constraints (|z|=4). "
        "Compute the corresponding maximum real part explicitly. Provide a final answer with verification of correctness and uniqueness if applicable."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_reflect_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking_2_2, answer_2_2]
    subtask_desc_3 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "SC_CoT | CoT"
    }
    thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_sc_instruction_3, 0, is_sub_task=True)
    agents.append(f"SC-CoT agent {cot_agent_3.id}, selecting and verifying maximizing θ, thinking: {thinking_3.content}; answer: {answer_3.content}")
    for i in range(N_reflect_3):
        feedback_3, correct_3 = await critic_agent_3([taskInfo, thinking_3, answer_3], "Please review and provide limitations of the final solution. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback: {feedback_3.content}; correct: {correct_3.content}")
        if correct_3.content == "True":
            break
        cot_inputs_3.extend([thinking_3, answer_3, feedback_3])
        thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_sc_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"SC-CoT agent {cot_agent_3.id}, refining final verification, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    return final_answer, logs
