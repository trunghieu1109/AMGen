async def forward_24(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Identify and verify that x, y, z are positive real numbers and that the given logarithmic equations "
        "log2(x/(yz))=1/2, log2(y/(xz))=1/3, log2(z/(xy))=1/4 are well-defined. Confirm domain constraints and extract expressions."
    )
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_1_1[i]([taskInfo], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, verifying positivity and domain, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_1.append(answer)
        possible_thinkings_1_1.append(thinking)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize verification of positivity and domain constraints.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    agents.append(f"Final Decision agent, stage_1.subtask_1, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_instruction_1_2 = (
        "Sub-task 2: Rewrite the given logarithmic equations in terms of a=log2(x), b=log2(y), c=log2(z). "
        "Express each as linear equations and validate correctness."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, rewriting equations, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_1_3 = (
        "Sub-task 3: Formulate the system of linear equations from the rewritten logarithmic equations. "
        "Write the system in matrix or standard linear form and check for solvability and independence."
    )
    cot_sc_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_1_3[i]([taskInfo, thinking_1_2, answer_1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_3[i].id}, formulating system, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_3.append(answer)
        possible_thinkings_1_3.append(thinking)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_answers_1_3 + possible_thinkings_1_3, "Sub-task 3: Synthesize system formulation and solvability.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    agents.append(f"Final Decision agent, stage_1.subtask_3, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 1: Solve the system of linear equations for a=log2(x), b=log2(y), c=log2(z). "
        "Use algebraic methods to find exact fractional values and verify solutions satisfy original equations."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_3, answer_1_3], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, solving system, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Compute the value of log2(x^4 y^3 z^2) = 4a + 3b + 2c using the solved values. "
        "Take absolute value and express as reduced fraction m/n with coprime positive integers."
    )
    cot_sc_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_2_2[i]([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_2[i].id}, computing expression, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2_2.append(answer)
        possible_thinkings_2_2.append(thinking)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_2_2 + possible_thinkings_2_2, "Sub-task 2: Synthesize computed value and fraction reduction.", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    agents.append(f"Final Decision agent, stage_2.subtask_2, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    debate_instruction_3_1 = (
        "Sub-task 1: Given the reduced fraction m/n from previous step, sum m + n. "
        "Verify fraction is in lowest terms and sum is correct. Provide final answer as requested."
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
                thinking, answer = await agent([taskInfo, thinking_2_2, answer_2_2], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_2, answer_2_2] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, summing fraction components, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_3_1[r].append(thinking)
            all_answer_3_1[r].append(answer)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + all_thinking_3_1[-1] + all_answer_3_1[-1], "Sub-task 1: Finalize sum m+n and verify fraction.", is_sub_task=True)
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    agents.append(f"Final Decision agent, stage_3.subtask_1, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    reflect_inst_3_2 = (
        "Sub-task 2: Final verification and reflection on the entire solution. "
        "Use all previous outputs (stage_1.subtask_1, stage_1.subtask_2, stage_1.subtask_3, stage_2.subtask_1, stage_2.subtask_2, stage_3.subtask_1). "
        "Verify that the values of a, b, c satisfy the original logarithmic equations exactly. "
        "Confirm the computed |log2(x^4 y^3 z^2)| matches the fraction m/n and the sum m+n. "
        "Cross-validate all intermediate results for consistency. If inconsistencies are found, flag explicitly and halt acceptance. "
        "Provide a detailed verification report alongside the final answer."
    )
    cot_reflect_instruction_3_2 = "Sub-task 2: Comprehensive final verification and reflection." + reflect_inst_3_2
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_2 = self.max_round
    cot_inputs_3_2 = [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2, thinking_1_3, answer_1_3, thinking_2_1, answer_2_1, thinking_2_2, answer_2_2, thinking_3_1, answer_3_1]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_reflect_instruction_3_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content, thinking_1_3.content, answer_1_3.content, thinking_2_1.content, answer_2_1.content, thinking_2_2.content, answer_2_2.content, thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, cot_reflect_instruction_3_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, verifying solution, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    for i in range(N_max_3_2):
        feedback, correct = await critic_agent_3_2([taskInfo, thinking_3_2, answer_3_2], "Please review and provide limitations or confirm correctness. Output 'True' if correct.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_2.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_2.extend([thinking_3_2, answer_3_2, feedback])
        thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, cot_reflect_instruction_3_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, refining solution, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Stage 3 Sub-task 2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
