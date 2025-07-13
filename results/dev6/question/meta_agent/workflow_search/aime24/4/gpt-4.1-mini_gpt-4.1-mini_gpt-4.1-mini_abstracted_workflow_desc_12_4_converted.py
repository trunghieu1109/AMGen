async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1_1 = (
        "Sub-task 1: Enumerate candidate primes p in ascending order, focusing on primes p congruent to 1 modulo 4. "
        "For each such prime p, iterate over all integers n in [1, p-1] to find all roots satisfying n^4 ≡ -1 (mod p). "
        "Provide a complete list of primes p and their corresponding roots modulo p. "
        "Prune primes without any roots to ensure only valid candidates proceed."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, enumerating primes and roots modulo p, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_instruction_1_2 = (
        "Sub-task 2: For each prime p and each root n modulo p found in Sub-task 1, apply Hensel's lemma to lift the solution to modulo p^2. "
        "Solve the linearized congruence f'(n)*k ≡ -f(n)/p (mod p), where f(x) = x^4 + 1, to find all possible lifts n + k*p modulo p^2. "
        "Handle cases where f'(n) ≡ 0 (mod p) carefully. Collect all lifted roots modulo p^2 for each prime p."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, lifting roots modulo p to modulo p^2, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = (
        "Sub-task 1: From the primes and lifted roots modulo p^2 obtained in Stage 1, identify the least prime p for which there exists at least one positive integer n such that p^2 divides n^4 + 1. "
        "Verify the existence of such n modulo p^2 and confirm minimality of p by ensuring no smaller prime satisfies the condition."
    )
    cot_sc_instruction_2_2 = (
        "Sub-task 2: For the identified least prime p, examine all lifted roots modulo p^2 to find the least positive integer m such that p^2 divides m^4 + 1. "
        "Compare all candidate m values and verify divisibility to ensure minimality."
    )
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    for i in range(self.max_sc):
        thinking_i, answer_i = await cot_agents_2_1[i]([taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, selecting least prime p, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)
    counter_2_1 = Counter([a.content for a in possible_answers_2_1])
    most_common_answer_2_1 = counter_2_1.most_common(1)[0][0]
    idx_2_1 = [a.content for a in possible_answers_2_1].index(most_common_answer_2_1)
    thinking_2_1 = possible_thinkings_2_1[idx_2_1]
    answer_2_1 = possible_answers_2_1[idx_2_1]
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, finding least m for selected p, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    debate_instruction_3_1 = (
        "Sub-task 1: Synthesize the results from Stage 2 to produce the final answer: the least prime p and the least positive integer m such that p^2 divides m^4 + 1. "
        "Present a clear, concise statement of the solution and confirm all conditions are met."
    )
    debate_instruction_3_2 = (
        "Sub-task 2: Verify the final solution by rechecking divisibility and minimality constraints. "
        "Reflect on the entire reasoning process to identify any overlooked edge cases or exceptions. Provide a final confirmation of correctness and completeness."
    )
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "Debate"
    }
    all_thinking_3_1 = []
    all_answer_3_1 = []
    for i, agent in enumerate(debate_agents_3_1):
        thinking_i, answer_i = await agent([taskInfo, thinking_2_2, answer_2_2], debate_instruction_3_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, synthesizing final answer, thinking: {thinking_i.content}; answer: {answer_i.content}")
        all_thinking_3_1.append(thinking_i)
        all_answer_3_1.append(answer_i)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + all_thinking_3_1 + all_answer_3_1, "Sub-task 3.1: Synthesize and finalize answer. Given all above, provide final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing final answer, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instruction_3_2,
        "context": ["user query", thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "Reflexion"
    }
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 3.2: Verify and reflect on final solution correctness and minimality." + reflect_inst
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_3_2 = [taskInfo, thinking_3_1, answer_3_1]
    thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, verifying final solution, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_3_2([taskInfo, thinking_3_2, answer_3_2], "Please review and provide limitations of provided solution. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_2.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_2.extend([thinking_3_2, answer_3_2, feedback])
        thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, refining solution, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
