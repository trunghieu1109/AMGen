async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Foundational definitions and prime condition

    # Subtask 1: Define divisibility condition and problem domain (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the divisibility condition p^2 divides n^4 + 1 as the congruence n^4 ≡ -1 (mod p^2). "
        "Clearly state that p is a prime number, n is a positive integer, and minimality conditions apply to p and n. "
        "Avoid assumptions about bounds or uniqueness. Provide a precise problem statement with definitions. "
        "Show all reasoning steps explicitly."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, defining problem domain, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Establish necessary condition p ≡ 1 (mod 8) for solutions modulo p (SC_CoT)
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Based on the formal problem definition, prove that for a prime p to have a solution to x^4 ≡ -1 (mod p), "
        "it is necessary that p ≡ 1 (mod 8). Explain the reasoning step-by-step using quartic residue theory and modular arithmetic. "
        "Show all intermediate steps and do not skip details."
    )
    N_sc = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, proving necessary condition p ≡ 1 (mod 8), thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2(
        [taskInfo] + possible_answers_0_2 + possible_thinkings_0_2,
        "Sub-task 2: Synthesize and choose the most consistent and correct proof that p ≡ 1 (mod 8) is necessary.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    # Subtask 3: Enumerate candidate primes p and find least p with solution x^4 ≡ -1 (mod p) (CoT)
    cot_instruction_0_3 = (
        "Sub-task 3: Enumerate primes p in ascending order that satisfy p ≡ 1 (mod 8). For each p, perform an exhaustive search for x in [1, p-1] such that x^4 ≡ -1 (mod p). "
        "Show detailed modular exponentiation steps for each candidate x and verify the congruence explicitly. Identify the least such prime p. "
        "Provide all arithmetic details and intermediate results."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_2, answer_0_2], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, enumerating primes and solutions modulo p, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 3: ", sub_tasks[-1])

    # Subtask 4: For identified prime p, verify existence of solutions modulo p^2 by Hensel lifting (Reflexion)
    reflexion_instruction_0_4 = (
        "Sub-task 4: For the prime p identified in Sub-task 3, verify the existence of solutions to x^4 ≡ -1 (mod p^2) by lifting each solution modulo p to modulo p^2 using Hensel's lemma. "
        "Explicitly compute f(x_0) = x_0^4 + 1 mod p^2, f'(x_0) = 4 x_0^3 mod p, the modular inverse of f'(x_0) mod p, the correction term, and the lifted root modulo p^2. "
        "Show all intermediate arithmetic steps in detail and verify correctness."
    )
    cot_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_0_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_0_4 = self.max_round
    cot_inputs_0_4 = [taskInfo, thinking_0_3, answer_0_3]
    subtask_desc_0_4 = {
        "subtask_id": "stage_0.subtask_4",
        "instruction": reflexion_instruction_0_4,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_0_4, answer_0_4 = await cot_agent_0_4(cot_inputs_0_4, reflexion_instruction_0_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_0_4.id}, performing Hensel lifting, thinking: {thinking_0_4.content}; answer: {answer_0_4.content}")
    for i in range(N_max_0_4):
        feedback_0_4, correct_0_4 = await critic_agent_0_4([taskInfo, thinking_0_4, answer_0_4],
            "Please review the above Hensel lifting computations and verify all arithmetic explicitly. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_0_4.id}, feedback: {feedback_0_4.content}; correctness: {correct_0_4.content}")
        if correct_0_4.content == "True":
            break
        cot_inputs_0_4.extend([thinking_0_4, answer_0_4, feedback_0_4])
        thinking_0_4, answer_0_4 = await cot_agent_0_4(cot_inputs_0_4, reflexion_instruction_0_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_0_4.id}, refining Hensel lifting, thinking: {thinking_0_4.content}; answer: {answer_0_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 1: Enumerate and lift all solutions modulo p and p^2

    # Subtask 1: Enumerate all solutions x modulo p to x^4 ≡ -1 (mod p) (SC_CoT)
    cot_sc_instruction_1_1 = (
        "Sub-task 1: For the prime p identified in Stage 0 Sub-task 4, enumerate all solutions x in [0, p-1] to x^4 ≡ -1 (mod p). "
        "Perform exhaustive search with detailed modular exponentiation steps for each candidate x. Provide a complete list of solutions with explicit verification."
    )
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_4, answer_0_4], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, enumerating solutions modulo p, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo] + possible_answers_1_1 + possible_thinkings_1_1,
        "Sub-task 1: Synthesize and list all solutions modulo p to x^4 ≡ -1 (mod p) with explicit verification.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 5: ", sub_tasks[-1])

    # Subtask 2: For each solution modulo p, perform Hensel lifting to modulo p^2 (CoT)
    cot_instruction_1_2 = (
        "Sub-task 2: For each solution x_0 modulo p found in Sub-task 1, perform Hensel lifting to find corresponding solutions modulo p^2. "
        "Explicitly compute f(x_0), f'(x_0), modular inverse of f'(x_0) mod p, correction term, and lifted root modulo p^2. "
        "Show all arithmetic steps and verify correctness for each lifted solution. Enumerate all distinct lifted solutions modulo p^2."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1, thinking_0_4, answer_0_4], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, performing Hensel lifting for all solutions modulo p, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 6: ", sub_tasks[-1])

    # Subtask 3: Exhaustive search modulo p^2 for additional solutions not lifts of modulo p (Reflexion)
    reflexion_instruction_1_3 = (
        "Sub-task 3: Perform an exhaustive search for all x in [1, p^2 - 1] to find solutions to x^4 ≡ -1 (mod p^2) that are not lifts of solutions modulo p. "
        "For each candidate x, compute x^4 + 1 modulo p^2 with detailed arithmetic steps. Record all solutions found. "
        "This ensures completeness of the solution set modulo p^2."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking_1_2, answer_1_2]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": reflexion_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, reflexion_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, exhaustive search modulo p^2, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(N_max_1_3):
        feedback_1_3, correct_1_3 = await critic_agent_1_3([taskInfo, thinking_1_3, answer_1_3],
            "Please verify all modular exponentiation and arithmetic steps explicitly. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, feedback: {feedback_1_3.content}; correctness: {correct_1_3.content}")
        if correct_1_3.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, answer_1_3, feedback_1_3])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, reflexion_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining exhaustive search, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 7: ", sub_tasks[-1])

    # Stage 2: Identify minimal positive integer m modulo p^2

    # Subtask 1: From all solutions modulo p^2, find least positive integer m with m^4 + 1 divisible by p^2 (Reflexion)
    reflexion_instruction_2_1 = (
        "Sub-task 1: From the complete set of solutions modulo p^2 found in Stage 1 Sub-task 3, identify the least positive integer m such that m^4 + 1 is divisible by p^2. "
        "Explicitly compute m^4 + 1 modulo p^2 with detailed arithmetic steps to verify divisibility."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking_1_3, answer_1_3]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": reflexion_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, reflexion_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, finding minimal m, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(N_max_2_1):
        feedback_2_1, correct_2_1 = await critic_agent_2_1([taskInfo, thinking_2_1, answer_2_1],
            "Please verify all modular exponentiation and minimality checks explicitly. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, feedback: {feedback_2_1.content}; correctness: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, answer_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, reflexion_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining minimal m, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 8: ", sub_tasks[-1])

    # Subtask 2: Verify minimality of m by checking all positive integers k < m (SC_CoT)
    cot_sc_instruction_2_2 = (
        "Sub-task 2: Verify the minimality of the integer m found in Sub-task 1 by checking all positive integers k with 1 ≤ k < m for the divisibility condition p^2 | k^4 + 1. "
        "For each k, compute k^4 + 1 modulo p^2 with explicit arithmetic steps and confirm no smaller integer satisfies the condition. "
        "Provide detailed verification and final confirmation of minimality."
    )
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_2_2[i]([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, verifying minimality of m, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_2.append(answer_i)
        possible_thinkings_2_2.append(thinking_i)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2(
        [taskInfo] + possible_answers_2_2 + possible_thinkings_2_2,
        "Sub-task 2: Synthesize and confirm minimality of m with explicit verification.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
