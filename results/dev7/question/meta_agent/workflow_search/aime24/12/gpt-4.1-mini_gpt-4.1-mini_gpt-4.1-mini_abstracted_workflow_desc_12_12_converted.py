async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = "Sub-task 1: Determine the set of valid complex numbers z such that |z|=4. Represent z in polar form as z = 4e^{iθ} with θ in [0, 2π). Explain why this parametrization covers all valid z and no other constraints limit z." 
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, parametrizing z on |z|=4, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)

    cot_instruction_1 = "Sub-task 1: Derive an explicit expression for the real part of (75 + 117i)z + (96 + 144i)/z in terms of θ, using z = 4e^{iθ}. Perform complex multiplication and division, separate real and imaginary parts, and simplify to a trigonometric form involving cos(θ) and sin(θ)."
    cot_sc_instruction_1 = "Sub-task 2: Validate the derived trigonometric expression for the real part by checking special cases θ=0 and θ=π/2, ensuring consistency with the original expression."

    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agents_sc_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]

    subtask_desc_1_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query", sub_tasks[-1]],
        "agent_collaboration": "CoT | SC_CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1([taskInfo, thinking_0, answer_0], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, deriving real part expression, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")

    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    for i in range(self.max_sc):
        thinking_i, answer_i = await cot_agents_sc_1[i]([taskInfo, thinking_0, answer_0, thinking_1_1, answer_1_1], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc_1[i].id}, validating expression, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)

    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent and correct trigonometric expression for the real part.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing validation, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")

    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    subtask_desc_1_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_1_1, answer_1_1],
        "agent_collaboration": "SC_CoT"
    }
    logs.append(subtask_desc_1_2)

    cot_sc_instruction_2_1 = "Sub-task 1: Identify the values of θ in [0, 2π) that maximize the derived trigonometric expression for the real part. Use trigonometric identities or calculus to find critical points and determine the global maximum."
    cot_instruction_2_2 = "Sub-task 2: Verify that the maximizing θ values correspond to valid z on |z|=4 and confirm the maximum real part is consistent with problem constraints."

    cot_agents_sc_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    subtask_desc_2_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_2, answer_1_2],
        "agent_collaboration": "SC_CoT | CoT"
    }

    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    for i in range(self.max_sc):
        thinking_i, answer_i = await cot_agents_sc_2_1[i]([taskInfo, thinking_1_2, answer_1_2], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc_2_1[i].id}, finding max θ, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)

    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_answers_2_1 + possible_thinkings_2_1, "Sub-task 1: Synthesize and choose the most consistent maximizing θ and maximum real part.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing max θ, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")

    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await cot_agent_2_2([taskInfo, thinking_2_1, answer_2_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, verifying max θ and max real part, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")

    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")

    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    subtask_desc_2_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking_2_1, answer_2_1],
        "agent_collaboration": "SC_CoT"
    }
    logs.append(subtask_desc_2_2)

    debate_instr_3_1 = "Sub-task 1: Decompose the final maximum real part into components if needed, simplify the numeric result, and present the final answer clearly. Provide a concise summary of the solution process and the maximum real part found."
    reflect_instr_3_2 = "Sub-task 2: Perform a final verification and reflection on the entire solution process, checking for logical consistency, correctness of assumptions, and completeness of the answer. Confirm the maximum real part is indeed the largest possible and all steps align with the problem statement."

    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)

    subtask_desc_3_1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_3_1,
        "context": ["user query", thinking_2_2, answer_2_2],
        "agent_collaboration": "Debate | Reflexion"
    }

    all_thinking_3_1 = [[] for _ in range(self.max_round)]
    all_answer_3_1 = [[] for _ in range(self.max_round)]

    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_2_2, answer_2_2], debate_instr_3_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_2, answer_2_2] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instr_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, decomposing and simplifying final answer, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_3_1[r].append(thinking_i)
            all_answer_3_1[r].append(answer_i)

    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + all_thinking_3_1[-1] + all_answer_3_1[-1], "Sub-task 1: Provide final simplified maximum real part and summary.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing answer, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")

    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)

    cot_reflect_instruction_3_2 = "Sub-task 2: Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_3_2 = [taskInfo, thinking_3_1, answer_3_1]

    subtask_desc_3_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_reflect_instruction_3_2,
        "context": ["user query", thinking_3_1, answer_3_1],
        "agent_collaboration": "Reflexion"
    }

    thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, cot_reflect_instruction_3_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, reflecting on final answer, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent_3_2([taskInfo, thinking_3_2, answer_3_2], "Please review and provide limitations or confirm correctness. Output exactly 'True' if correct.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_2.extend([thinking_3_2, answer_3_2, feedback])
        thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, cot_reflect_instruction_3_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, refining answer, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")

    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    for i, st in enumerate(sub_tasks):
        print(f"Step {i}: ", st)
    return final_answer, logs
