async def forward_24(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    # Stage 0: select_and_verify_elements_under_constraints
    cot_instruction_0 = (
        "Sub-task 1: Identify and verify the given variables and constraints: x, y, z are positive real numbers. "
        "Confirm the given logarithmic equations and the base of the logarithm (base 2). Ensure the problem is well-defined with no ambiguity in domain or notation. "
        "Confirm the target expression |log2(x^4 y^3 z^2)| and the requirement to express it as a reduced fraction m/n with coprime positive integers, then find m+n. "
        "Avoid assumptions about multiple solutions or negative values since variables are positive."
    )
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, verifying problem setup, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Stage 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    # Stage 1: derive_and_validate_representations
    cot_instruction_1_1 = (
        "Sub-task 1: Rewrite the given logarithmic equations into linear equations in terms of a=log2(x), b=log2(y), c=log2(z). "
        "Use logarithm properties to express each equation as a linear combination of a, b, and c equal to the given constants. "
        "Carefully handle signs and coefficients to avoid errors."
    )
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent([taskInfo, thinking_0, answer_0], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, rewriting equations, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Stage 1 subtask 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_instruction_1_2 = (
        "Sub-task 2: Formulate the system of linear equations from the expressions obtained in subtask_1. "
        "Represent the system in matrix or equation form suitable for solving. Verify the system is consistent and has a unique solution given the positivity constraints."
    )
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_0.content, answer_0.content, thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent([taskInfo, thinking_0, answer_0, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formulating linear system, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Stage 1 subtask 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Stage 2: infer_compute_parameters_from_composite_data
    cot_sc_instruction_2_1 = (
        "Sub-task 1: Solve the linear system for a=log2(x), b=log2(y), c=log2(z) using appropriate algebraic methods. "
        "Ensure the solution satisfies all equations and is consistent with positivity of x, y, z."
    )
    N = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_0.content, answer_0.content, thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_i, answer_i = await cot_agents_2_1[i]([taskInfo, thinking_0, answer_0, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, solving linear system, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)

    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_answers_2_1 + possible_thinkings_2_1, "Sub-task 1: Synthesize and choose the most consistent and correct solution for a, b, c.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing solution for a,b,c, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Stage 2 subtask 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Compute the value of 4a + 3b + 2c using the solved values of a, b, and c. "
        "This corresponds to log2(x^4 y^3 z^2)."
    )
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_i, answer_i = await cot_agents_2_2[i]([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, computing 4a+3b+2c, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_2.append(answer_i)
        possible_thinkings_2_2.append(thinking_i)

    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_2_2 + possible_thinkings_2_2, "Sub-task 2: Synthesize and choose the most consistent and correct value for 4a+3b+2c.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing value of 4a+3b+2c, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Stage 2 subtask 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Stage 3: decompose_simplify_and_sum_components
    debate_instruction_3_1 = (
        "Sub-task 1: Calculate the absolute value of the result from stage_2.subtask_2. "
        "Express this absolute value as a reduced fraction m/n where m and n are coprime positive integers. Simplify the fraction fully."
    )
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_2_2, answer_2_2], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_2, answer_2_2] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying fraction, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_3_1[r].append(thinking_i)
            all_answer_3_1[r].append(answer_i)

    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + all_thinking_3_1[-1] + all_answer_3_1[-1], "Sub-task 1: Finalize simplified fraction and absolute value.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing fraction simplification, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Stage 3 subtask 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    debate_instruction_3_2 = (
        "Sub-task 2: Find the sum m + n from the simplified fraction obtained in subtask_1. "
        "Verify the correctness of the simplification and final sum. Provide the final answer to the original problem."
    )
    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_2 = self.max_round
    all_thinking_3_2 = [[] for _ in range(N_max_3_2)]
    all_answer_3_2 = [[] for _ in range(N_max_3_2)]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instruction_3_2,
        "context": ["user query", thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_2):
        for i, agent in enumerate(debate_agents_3_2):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_3_1, answer_3_1], debate_instruction_3_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_3_1, answer_3_1] + all_thinking_3_2[r-1] + all_answer_3_2[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, summing m+n, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_3_2[r].append(thinking_i)
            all_answer_3_2[r].append(answer_i)

    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_2, answer_3_2 = await final_decision_agent_3_2([taskInfo] + all_thinking_3_2[-1] + all_answer_3_2[-1], "Sub-task 2: Finalize sum m+n and verify correctness.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing sum m+n, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Stage 3 subtask 2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
