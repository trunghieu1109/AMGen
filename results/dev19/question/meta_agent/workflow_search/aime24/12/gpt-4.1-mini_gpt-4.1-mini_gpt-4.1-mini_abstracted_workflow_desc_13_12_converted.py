async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Parameterize z, rewrite expression, extract explicit real part (SC_CoT)
    cot_sc_instruction_0_1 = (
        "Sub-task 1: Express the complex variable z in polar form as z = 4e^{iθ} with θ in [0, 2π), "
        "explicitly stating the constraint |z|=4 and the parameterization of z by θ. Avoid any simplification or substitution beyond this parameterization."
    )
    cot_sc_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_1[i].id}, parameterize z, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + possible_thinkings_0_1, "Sub-task 1: Synthesize and choose the most consistent parameterization of z.", is_sub_task=True)
    sub_tasks.append(f"Stage 0 Subtask 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Rewrite the expression (75+117i)z + (96+144i)/z in terms of θ using the polar form of z from subtask_1, "
        "carefully simplifying the division by z as multiplication by its conjugate form. Present the expression explicitly in terms of cosθ and sinθ without combining or optimizing terms."
    )
    cot_sc_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_2[i]([taskInfo, thinking_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_2[i].id}, rewrite expression in cosθ and sinθ, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_2.append(answer)
        possible_thinkings_0_2.append(thinking)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent rewritten expression.", is_sub_task=True)
    sub_tasks.append(f"Stage 0 Subtask 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    cot_sc_instruction_0_3 = (
        "Sub-task 3: Derive the explicit expression for the real part of (75+117i)z + (96+144i)/z as a function of θ, "
        "carefully separating real and imaginary components. The output must be the exact linear combination of cosine and sine terms with explicit coefficients (e.g., Real part = 396 cosθ − 324 sinθ). "
        "Do not attempt any further combination or optimization at this stage."
    )
    cot_sc_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_3[i]([taskInfo, thinking_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_3[i].id}, derive explicit real part expression, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_3.append(answer)
        possible_thinkings_0_3.append(thinking)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent explicit real part expression.", is_sub_task=True)
    sub_tasks.append(f"Stage 0 Subtask 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1: Compute magnitudes and arguments (Debate), combine into single cosine (Debate), verify combined expression (SC_CoT)
    debate_instruction_1_1 = (
        "Sub-task 1: Compute the magnitudes and arguments (angles) of the complex coefficients 75+117i and 96+144i precisely, "
        "showing all steps and verifying calculations. These values will support the trigonometric manipulation in subsequent subtasks. "
        "Do not use these values to alter the explicit real part expression derived in stage_0.subtask_3."
    )
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_max_1_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1_1)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_0_3], debate_instruction_1_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0_3] + all_thinking_1_1[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, compute magnitudes and arguments, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_1[r].append(thinking)
            all_answer_1_1[r].append(answer)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + all_thinking_1_1[-1], "Sub-task 1: Final decision on magnitudes and arguments.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    debate_instruction_1_2 = (
        "Sub-task 2: Using the exact explicit real part expression from stage_0.subtask_3 (396 cosθ − 324 sinθ), "
        "convert the sine term into a cosine term using the identity sinθ = cos(θ − π/2). Then, combine the resulting expression into a single cosine function of the form R cos(θ + φ). "
        "Perform this in two clear steps: (a) rewrite the expression as A cosθ + B cos(θ − π/2) with A=396 and B=−324, "
        "(b) combine into R cos(θ + φ) by computing R = √(A² + B²) and φ = arctan(B/A). Explicitly verify each arithmetic step and do not recalculate or alter the original coefficients. "
        "Provide detailed reasoning and intermediate results."
    )
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_1_2 = self.max_round
    all_thinking_1_2 = [[] for _ in range(N_max_1_2)]
    all_answer_1_2 = [[] for _ in range(N_max_1_2)]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_1_2,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_2):
        for i, agent in enumerate(debate_agents_1_2):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_0_3], debate_instruction_1_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0_3] + all_thinking_1_2[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, combine into single cosine, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_2[r].append(thinking)
            all_answer_1_2[r].append(answer)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + all_thinking_1_2[-1], "Sub-task 2: Final decision on combined single cosine expression.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_1_3 = (
        "Sub-task 3: Verify that the combined single cosine expression R cos(θ + φ) exactly matches the explicit real part expression 396 cosθ − 324 sinθ from stage_0.subtask_3 by expanding and comparing coefficients. "
        "If discrepancies are found, halt and report errors. This verification must be rigorous and documented step-by-step to ensure no arithmetic or transcription errors occurred in the combination step."
    )
    cot_sc_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_0_3.content, thinking_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_1_3[i]([taskInfo, thinking_0_3, thinking_1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_3[i].id}, verify combined expression, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_3.append(answer)
        possible_thinkings_1_3.append(thinking)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, "Sub-task 3: Final verification of combined expression.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    # Stage 2: Maximize the real part (SC_CoT)
    cot_sc_instruction_2_1 = (
        "Sub-task 1: Formulate the problem of maximizing the real part as maximizing R cos(θ + φ) over θ in [0, 2π), "
        "where R and φ are from the verified combined expression in stage_1.subtask_3. Identify the maximum value of the real part (which is R) and the corresponding θ that achieves it. "
        "Provide a clear explanation of why the maximum is R and how the maximizing θ is determined."
    )
    cot_sc_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_2_1[i]([taskInfo, thinking_1_3], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_1[i].id}, maximize real part, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2_1.append(answer)
        possible_thinkings_2_1.append(thinking)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_thinkings_2_1, "Sub-task 1: Final decision on maximum real part.", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Subtask 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Stage 3: Verify maximum respects constraint and state final answer (SC_CoT and CoT)
    cot_sc_instruction_3_1 = (
        "Sub-task 1: Verify that the maximum real part value found in stage_2.subtask_1 respects the constraint |z|=4 and confirm the uniqueness or multiplicity of maximizing θ values within the domain [0, 2π). "
        "Provide a detailed justification based on the periodicity and properties of cosine functions."
    )
    cot_sc_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_3_1[i]([taskInfo, thinking_2_1], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3_1[i].id}, verify max respects constraint, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_3_1.append(answer)
        possible_thinkings_3_1.append(thinking)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + possible_thinkings_3_1, "Sub-task 1: Final decision on verification of maximum.", is_sub_task=True)
    sub_tasks.append(f"Stage 3 Subtask 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    cot_instruction_3_2 = (
        "Sub-task 2: State the final largest possible real part of the expression (75+117i)z + (96+144i)/z under the constraint |z|=4, "
        "summarizing the solution clearly and concisely. Do not introduce new calculations or assumptions. Reference the verified maximum value and corresponding θ from previous subtasks."
    )
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_instruction_3_2,
        "context": ["user query", thinking_3_1.content, answer_2_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_3_2, answer_3_2 = await cot_agent_3_2([taskInfo, thinking_3_1, answer_2_1], cot_instruction_3_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_2.id}, state final answer, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Stage 3 Subtask 2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
